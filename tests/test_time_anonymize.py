"""时间匿名化接口 POST /api/v1/time-anonymize 的端到端测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/time-anonymize"


def make_record(obs_id: str, observed_at: str) -> dict:
    return {"id": obs_id, "observed_at": observed_at}


def payload(k: int = 2, max_window_hours: int = 24, records=None) -> dict:
    return {
        "k": k,
        "max_window_hours": max_window_hours,
        "records": records if records is not None else [],
    }


# ---------------------------------------------------------------- 基本功能

def test_basic_partition_with_gap_and_input_order():
    # UTC 0 点 2 条，1、2 点各 1 条，10、11 点各 1 条；记录输入顺序打乱。
    records = [
        make_record("obs-01", "2026-09-25T00:20:00Z"),
        make_record("obs-06", "2026-09-25T11:40:00Z"),
        make_record("obs-02", "2026-09-25T01:05:00Z"),
        make_record("obs-05", "2026-09-25T10:50:00Z"),
        make_record("obs-03", "2026-09-25T02:59:00Z"),
        make_record("obs-04", "2026-09-25T00:45:00Z"),  # 与 obs-01 同小时桶
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=3, records=records))
    assert response.status_code == 200
    body = response.json()

    assert body["k"] == 2
    assert body["max_window_hours"] == 3
    assert body["total_records"] == 6

    # 三窗：[0,1) 2 条（代价 2）、[1,3) 2 条（跨度 2，代价 4）、
    # [10,12) 2 条（跨度 2，代价 4），Σ=10，为全局最小。
    assert body["windows"] == [
        {"start": "2026-09-25T00:00:00Z", "end": "2026-09-25T01:00:00Z", "count": 2},
        {"start": "2026-09-25T01:00:00Z", "end": "2026-09-25T03:00:00Z", "count": 2},
        {"start": "2026-09-25T10:00:00Z", "end": "2026-09-25T12:00:00Z", "count": 2},
    ]

    # 逐记录结果与输入等长、等序；同小时桶记录不可拆开。
    assert [r["id"] for r in body["records"]] == [
        "obs-01", "obs-06", "obs-02", "obs-05", "obs-03", "obs-04",
    ]
    expected = {
        "obs-01": ("2026-09-25T00:00:00Z", "2026-09-25T01:00:00Z"),
        "obs-04": ("2026-09-25T00:00:00Z", "2026-09-25T01:00:00Z"),
        "obs-02": ("2026-09-25T01:00:00Z", "2026-09-25T03:00:00Z"),
        "obs-03": ("2026-09-25T01:00:00Z", "2026-09-25T03:00:00Z"),
        "obs-05": ("2026-09-25T10:00:00Z", "2026-09-25T12:00:00Z"),
        "obs-06": ("2026-09-25T10:00:00Z", "2026-09-25T12:00:00Z"),
    }
    for entry in body["records"]:
        assert set(entry) == {"id", "window_start", "window_end"}
        assert (entry["window_start"], entry["window_end"]) == expected[entry["id"]]

    # 不回传原时刻或其他原字段。
    assert "2026-09-25T00:20:00Z" not in response.text
    assert "observed_at" not in response.text


def test_timezone_offset_converts_to_utc_and_buckets():
    # +08:00 凌晨 08:20 -> UTC 00:20；+08:00 上午 11:20 -> UTC 03:20。
    records = [
        make_record("obs-a", "2026-09-25T08:20:00+08:00"),
        make_record("obs-b", "2026-09-25T11:20:00+08:00"),
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=24, records=records))
    assert response.status_code == 200
    body = response.json()
    assert body["windows"] == [
        {"start": "2026-09-25T00:00:00Z", "end": "2026-09-25T04:00:00Z", "count": 2},
    ]


def test_named_timezone_in_brackets_is_supported():
    records = [
        make_record("obs-a", "2026-09-25T08:20:00[Asia/Shanghai]"),
        make_record("obs-b", "2026-09-25T11:20:00[Asia/Shanghai]"),
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=24, records=records))
    assert response.status_code == 200
    body = response.json()
    assert body["windows"] == [
        {"start": "2026-09-25T00:00:00Z", "end": "2026-09-25T04:00:00Z", "count": 2},
    ]


def test_offset_crossing_midnight_is_same_utc_day_valid():
    # UTC 23:30 与 00:10 本属不同 UTC 日期；这里取同在 UTC 9/25 但本地跨日：
    # 东十区 2026-09-26T09:30 -> UTC 2026-09-25T23:30；
    # 北京时间 2026-09-25T08:10 -> UTC 2026-09-25T00:10。同一 UTC 日期，合法。
    records = [
        make_record("obs-a", "2026-09-26T09:30:00+10:00"),
        make_record("obs-b", "2026-09-25T08:10:00+08:00"),
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=24, records=records))
    assert response.status_code == 200
    body = response.json()
    assert body["windows"] == [
        {"start": "2026-09-25T00:00:00Z", "end": "2026-09-26T00:00:00Z", "count": 2},
    ]


def test_k_one_each_occupied_hour_is_its_own_window():
    records = [
        make_record("obs-a", "2026-09-25T05:10:00Z"),
        make_record("obs-b", "2026-09-25T09:50:00Z"),
    ]
    response = client.post(ENDPOINT, json=payload(k=1, max_window_hours=24, records=records))
    assert response.status_code == 200
    body = response.json()
    # 每窗跨度 1、记录数 1，各自成窗；合窗代价 10 大于 2。
    assert body["windows"] == [
        {"start": "2026-09-25T05:00:00Z", "end": "2026-09-25T06:00:00Z", "count": 1},
        {"start": "2026-09-25T09:00:00Z", "end": "2026-09-25T10:00:00Z", "count": 1},
    ]
    assert body["records"][0]["window_start"] == "2026-09-25T05:00:00Z"
    assert body["records"][1]["window_start"] == "2026-09-25T09:00:00Z"


def test_extra_fields_are_accepted_but_not_returned():
    # 与 geo 接口一致：多余原字段不参与处理，也不能回传。
    records = [
        {**make_record("obs-a", "2026-09-25T01:00:00Z"), "species": "金丝猴",
         "observer": "志愿者A"},
        make_record("obs-b", "2026-09-25T01:30:00Z"),
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=24, records=records))
    assert response.status_code == 200
    assert "金丝猴" not in response.text
    assert "observer" not in response.text
    assert "志愿者A" not in response.text


# ---------------------------------------------------------------- 全局最优与字典序裁决

def test_tie_break_lexicographic_end_hours():
    """桶 0..4 的记录数为 [2,1,2,1,2]，k=3。

    - 0、2、4 单桶均不足 3，无法单独成窗；
    - [0,1]+[2,3,4]：3×2 + 5×3 = 21，结束小时序列 (1,4)；
    - [0,1,2]+[3,4]：5×3 + 3×2 = 21，结束小时序列 (2,4)；
    - 整窗 [0..4]：8×5 = 40，代价更大。
    总代价并列时按结束小时序列字典序取最早者 (1,4)。
    """
    records = []
    for hour, count in enumerate((2, 1, 2, 1, 2)):
        for serial in range(count):
            records.append(make_record(f"h{hour}-{serial}", f"2026-09-25T{hour:02d}:00:00Z"))
    response = client.post(ENDPOINT, json=payload(k=3, max_window_hours=24, records=records))
    assert response.status_code == 200
    body = response.json()
    assert body["windows"] == [
        {"start": "2026-09-25T00:00:00Z", "end": "2026-09-25T02:00:00Z", "count": 3},
        {"start": "2026-09-25T02:00:00Z", "end": "2026-09-25T05:00:00Z", "count": 5},
    ]


def test_gap_hours_count_toward_span_cost():
    # 桶 0 有 2 条、桶 3 有 2 条，k=2：分窗各代价 2（合计 4），
    # 合窗记录数 4、跨度 4，代价 16。max_window_hours=3 时合窗不可行，分窗可行。
    records = [
        make_record("obs-a", "2026-09-25T00:00:00Z"),
        make_record("obs-b", "2026-09-25T00:30:00Z"),
        make_record("obs-c", "2026-09-25T03:00:00Z"),
        make_record("obs-d", "2026-09-25T03:30:00Z"),
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=3, records=records))
    assert response.status_code == 200
    body = response.json()
    assert body["windows"] == [
        {"start": "2026-09-25T00:00:00Z", "end": "2026-09-25T01:00:00Z", "count": 2},
        {"start": "2026-09-25T03:00:00Z", "end": "2026-09-25T04:00:00Z", "count": 2},
    ]

    # max_window_hours=4 合窗可行，但代价更大，仍应分窗。
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=4, records=records))
    assert response.status_code == 200
    body = response.json()
    assert len(body["windows"]) == 2


def test_must_merge_when_single_hour_below_k():
    # 桶 5 有 1 条、桶 6 有 2 条，k=2：桶 5 无法单独成窗，必须并入桶 6。
    records = [
        make_record("obs-a", "2026-09-25T05:59:00Z"),
        make_record("obs-b", "2026-09-25T06:00:00Z"),
        make_record("obs-c", "2026-09-25T06:30:00Z"),
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=24, records=records))
    assert response.status_code == 200
    body = response.json()
    assert body["windows"] == [
        {"start": "2026-09-25T05:00:00Z", "end": "2026-09-25T07:00:00Z", "count": 3},
    ]


def test_infeasible_when_span_limit_forces_a_window_over_limit():
    # 桶 0 有 1 条、桶 23 有 3 条，k=2：桶 0 必须并入后桶，
    # 覆盖跨度 24 小时；max_window_hours=23 不可行。
    records = [
        make_record("obs-a", "2026-09-25T00:10:00Z"),
        make_record("obs-b", "2026-09-25T23:10:00Z"),
        make_record("obs-c", "2026-09-25T23:20:00Z"),
        make_record("obs-d", "2026-09-25T23:30:00Z"),
    ]
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=23, records=records))
    assert response.status_code == 422
    assert "可行窗口划分" in response.json()["detail"]

    # 放宽到 24 小时后唯一可行：全部并入一窗。
    response = client.post(ENDPOINT, json=payload(k=2, max_window_hours=24, records=records))
    assert response.status_code == 200
    body = response.json()
    assert body["windows"] == [
        {"start": "2026-09-25T00:00:00Z", "end": "2026-09-26T00:00:00Z", "count": 4},
    ]


# ---------------------------------------------------------------- 422 校验

def test_422_when_total_less_than_k():
    records = [make_record("obs-a", "2026-09-25T01:00:00Z")]
    response = client.post(ENDPOINT, json=payload(k=2, records=records))
    assert response.status_code == 422
    assert "小于 k" in response.json()["detail"]


def test_422_on_empty_records():
    response = client.post(ENDPOINT, json=payload(k=1, records=[]))
    assert response.status_code == 422


def test_422_when_k_not_positive_integer():
    records = [make_record("obs-a", "2026-09-25T01:00:00Z")]
    for bad_k in (0, -2, True, "2", 2.5):
        response = client.post(ENDPOINT, json=payload(k=bad_k, records=records))
        assert response.status_code == 422


def test_422_when_max_window_hours_out_of_range():
    records = [make_record("obs-a", "2026-09-25T01:00:00Z")]
    for bad_hours in (0, -1, 25, 100):
        response = client.post(
            ENDPOINT, json=payload(k=1, max_window_hours=bad_hours, records=records)
        )
        assert response.status_code == 422
        assert "max_window_hours" in response.json()["detail"]


def test_422_when_max_window_hours_not_integer():
    records = [make_record("obs-a", "2026-09-25T01:00:00Z")]
    for bad_hours in (True, "3", 2.5):
        response = client.post(
            ENDPOINT, json=payload(k=1, max_window_hours=bad_hours, records=records)
        )
        assert response.status_code == 422


def test_422_when_id_invalid():
    variants = [
        [{"observed_at": "2026-09-25T01:00:00Z"}],  # 缺 id
        [make_record("", "2026-09-25T01:00:00Z")],  # 空 id
        [make_record("   ", "2026-09-25T01:00:00Z")],  # 空白 id
        [{"id": 123, "observed_at": "2026-09-25T01:00:00Z"}],  # 非字符串 id
    ]
    for records in variants:
        response = client.post(ENDPOINT, json=payload(k=1, records=records))
        assert response.status_code == 422
        assert "id" in response.json()["detail"]


def test_422_when_id_duplicated():
    records = [
        make_record("obs-a", "2026-09-25T01:00:00Z"),
        make_record("obs-a", "2026-09-25T02:00:00Z"),
    ]
    response = client.post(ENDPOINT, json=payload(k=1, records=records))
    assert response.status_code == 422
    assert "唯一" in response.json()["detail"]


def test_422_when_observed_at_missing_or_malformed():
    bad_values = (
        None,
        "",
        "   ",
        123,
        "2026-09-25 01:00:00",  # 合法但无时区
        "2026-09-25T01:00:00",  # 无时区
        "not-a-timestamp",
        "2026-13-25T01:00:00Z",
        "2026-09-25T25:00:00Z",
        ["2026-09-25T01:00:00Z"],
    )
    for bad_value in bad_values:
        records = [{"id": "obs-a", "observed_at": bad_value}]
        response = client.post(ENDPOINT, json=payload(k=1, records=records))
        assert response.status_code == 422, bad_value
        assert "observed_at" in response.json()["detail"], bad_value


def test_422_when_named_timezone_unknown():
    records = [make_record("obs-a", "2026-09-25T01:00:00[Not/A_Zone]")]
    response = client.post(ENDPOINT, json=payload(k=1, records=records))
    assert response.status_code == 422
    assert "时区" in response.json()["detail"]


def test_422_when_utc_dates_differ():
    records = [
        make_record("obs-a", "2026-09-25T23:59:00Z"),
        make_record("obs-b", "2026-09-26T00:00:00Z"),
    ]
    response = client.post(ENDPOINT, json=payload(k=1, records=records))
    assert response.status_code == 422
    assert "同一个 UTC 日期" in response.json()["detail"]


def test_422_on_missing_or_extra_body_fields():
    response = client.post(ENDPOINT, json={"k": 2, "records": []})
    assert response.status_code == 422

    records = [
        make_record("obs-a", "2026-09-25T01:00:00Z"),
        make_record("obs-b", "2026-09-25T01:30:00Z"),
    ]
    response = client.post(
        ENDPOINT, json={**payload(k=2, records=records), "extra": 1}
    )
    assert response.status_code == 422


def test_422_when_record_is_not_object():
    response = client.post(
        ENDPOINT, json=payload(k=1, records=[["not", "a", "dict"]])
    )
    assert response.status_code == 422
