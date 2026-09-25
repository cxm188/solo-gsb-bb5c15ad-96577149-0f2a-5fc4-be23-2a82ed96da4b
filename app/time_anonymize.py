"""时间匿名化：把同一 UTC 日期内带时区的观测时刻归入满足 k-匿名的连续 UTC 小时窗口。

记录时刻先统一换算为 UTC 并按整点向下归槽；只保留有记录的小时，再把这些
小时按时间顺序切分为互不重叠的连续窗口。每个窗口至少包含 k 条记录，窗口内
首个与末个有记录小时的覆盖跨度（含中间没有记录的空档小时）不超过
max_window_hours；同一小时内的记录必须整体进入同一个窗口，不能拆开。

在所有可行划分中最小化各窗“记录数 × 跨度小时”之和；总和相同时，按各窗口
结束小时序列的字典序选最早者。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .anonymizer import unprocessable

# max_window_hours 的允许范围。
MIN_WINDOW_HOURS_LIMIT = 1
MAX_WINDOW_HOURS_LIMIT = 24


def prepare_utc_hours(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[int], date]:
    """校验记录并返回 (id 列表, UTC 小时槽列表, UTC 日期)。

    两个列表等长且与输入同序；小时槽为该 UTC 日期内的整点序号 0～23
    （分钟、秒向下截断）。每条记录必须有唯一非空字符串 id 与带时区的
    ISO 8601 时刻 observed_at；全部时刻换算到 UTC 后必须落在同一个 UTC 日期。
    其余字段不参与处理，也不回传。
    """
    if not records:
        raise unprocessable("records 至少需要包含 1 条记录")

    ids: list[str] = []
    hours: list[int] = []
    seen_ids: set[str] = set()
    utc_date: date | None = None

    for index, record in enumerate(records):
        label = f"第 {index + 1} 条记录"

        obs_id = record.get("id")
        if not isinstance(obs_id, str) or not obs_id.strip():
            raise unprocessable(f"{label}的 id 必须是非空字符串")
        if obs_id in seen_ids:
            raise unprocessable(f"id {obs_id!r} 重复出现，每条记录的 id 必须唯一")
        seen_ids.add(obs_id)
        ids.append(obs_id)

        utc_dt = _parse_observed_at(record.get("observed_at"), label)
        if utc_date is None:
            utc_date = utc_dt.date()
        elif utc_dt.date() != utc_date:
            raise unprocessable(
                "所有记录换算到 UTC 后必须落在同一个 UTC 日期内："
                f"{obs_id!r} 位于 {utc_dt.date().isoformat()}，"
                f"此前记录位于 {utc_date.isoformat()}"
            )
        hours.append(utc_dt.hour)

    # 记录非空时 utc_date 必然已被赋值。
    assert utc_date is not None
    return ids, hours, utc_date


def _parse_observed_at(value: Any, label: str) -> datetime:
    """解析带时区的 ISO 8601 时刻并换算为 UTC（拒绝无时区的本地时间）。

    支持显式 UTC 偏移（如 ``Z``、``+08:00``）以及方括号包裹的 IANA 时区名
    （如 ``2026-09-25T08:00:00[Asia/Shanghai]``）。
    """
    if not isinstance(value, str) or not value.strip():
        raise unprocessable(
            f"{label}的 observed_at 必须是带时区的 ISO 8601 时间字符串"
        )

    text = value.strip()
    named_zone: str | None = None
    if text.endswith("]"):
        head, separator, zone = text.rpartition("[")
        named_zone = zone[:-1].strip()
        if not separator or not head or not named_zone:
            raise unprocessable(
                f"{label}的 observed_at 不是合法的 ISO 8601 时间: {value!r}"
            )
        text = head

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise unprocessable(
            f"{label}的 observed_at 不是合法的 ISO 8601 时间: {value!r}"
        )

    if named_zone is not None:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(named_zone))
        except (ZoneInfoNotFoundError, ValueError):
            raise unprocessable(
                f"{label}的 observed_at 含未知时区 {named_zone!r}"
            )

    if parsed.tzinfo is None:
        raise unprocessable(
            f"{label}的 observed_at 必须带时区（UTC 偏移量或 IANA 时区名），"
            "不接受无时区的本地时间"
        )

    return parsed.astimezone(timezone.utc)


def partition_windows(
    hours: list[int],
    k: int,
    max_window_hours: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    """求全局最优窗口划分。

    返回 (窗口列表, 每条记录所属窗口下标)：窗口为
    ``(首个有记录小时, 末个有记录小时)``（均含），按时间升序；记录归属列表
    与 ``hours`` 等长、同序。最小化 Σ 记录数×跨度，并列时按各窗结束小时
    序列的字典序取最早者；无可行划分时返回 422。
    """
    occupied = sorted(set(hours))
    m = len(occupied)
    position = {hour: i for i, hour in enumerate(occupied)}

    # 每个有记录小时桶内的记录数，以及前缀和（prefix[j]-prefix[i] 即桶
    # i..j-1 的记录总数）。
    bucket_count = [0] * m
    for hour in hours:
        bucket_count[position[hour]] += 1
    prefix = [0]
    for count in bucket_count:
        prefix.append(prefix[-1] + count)

    # best[j] = 只覆盖前 j 个有记录小时时的最优方案：
    # (总代价, 各窗结束小时序列, 最后一个切点 i)；None 表示不可达。
    # 结束小时取“末个有记录小时（含）”，用于字典序裁决；各元素统一 +1
    # （改成半开右端）不改变字典序比较结果。
    best: list[tuple[int, tuple[int, ...], int] | None] = [None] * (m + 1)
    best[0] = (0, (), 0)

    for j in range(1, m + 1):
        end_hour = occupied[j - 1]
        for i in range(j):
            if best[i] is None:
                continue
            start_hour = occupied[i]
            span = end_hour - start_hour + 1
            # 跨度按首末有记录小时计算，中间空档小时也计入覆盖跨度。
            if span > max_window_hours:
                continue
            count = prefix[j] - prefix[i]
            if count < k:
                continue
            prev_cost, prev_ends, _ = best[i]  # type: ignore[misc]
            cost = prev_cost + count * span
            ends = prev_ends + (end_hour,)
            current = best[j]
            if current is None or (cost, ends) < (current[0], current[1]):
                best[j] = (cost, ends, i)

    if best[m] is None:
        raise unprocessable(
            "不存在满足每窗至少 k 条、且首末有记录小时覆盖跨度不超过 "
            f"{max_window_hours} 小时的可行窗口划分"
        )

    # 回溯切点并重建窗口（时间升序）。
    cuts: list[tuple[int, int]] = []
    j = m
    while j > 0:
        _, _, i = best[j]  # type: ignore[misc]
        cuts.append((i, j))
        j = i
    cuts.reverse()
    windows = [(occupied[i], occupied[j - 1]) for i, j in cuts]

    # 每条记录按其小时桶找到所属窗口；同一小时桶整体归入同一窗口。
    window_of_bucket: dict[int, int] = {}
    for window_index, (i, j) in enumerate(cuts):
        for bucket in range(i, j):
            window_of_bucket[bucket] = window_index
    assignment = [window_of_bucket[position[hour]] for hour in hours]

    return windows, assignment


def _hour_iso(utc_date: date, hour: int) -> str:
    """UTC 日期内第 hour 个整点的 ISO 8601 字符串（hour 可取 24 表示次日 0 点）。"""
    instant = datetime.combine(utc_date, time.min, tzinfo=timezone.utc) + timedelta(
        hours=hour
    )
    return instant.isoformat().replace("+00:00", "Z")


def anonymize_times(
    records: list[dict[str, Any]],
    k: int,
    max_window_hours: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """时间匿名化主流程，返回 (逐记录结果, 各窗结果)。

    逐记录结果与输入等长、等序，只含 id 与所属窗口的半开 UTC 小时区间；
    各窗结果按时间升序给出区间与计数。不回传原时刻或其他原字段。
    """
    ids, hours, utc_date = prepare_utc_hours(records)
    windows, assignment = partition_windows(hours, k, max_window_hours)

    counts = [0] * len(windows)
    for window_index in assignment:
        counts[window_index] += 1

    record_results = [
        {
            "id": obs_id,
            "window_start": _hour_iso(utc_date, windows[window_index][0]),
            "window_end": _hour_iso(utc_date, windows[window_index][1] + 1),
        }
        for obs_id, window_index in zip(ids, assignment)
    ]
    window_results = [
        {
            "start": _hour_iso(utc_date, start_hour),
            "end": _hour_iso(utc_date, end_hour + 1),
            "count": counts[window_index],
        }
        for window_index, (start_hour, end_hour) in enumerate(windows)
    ]
    return record_results, window_results
