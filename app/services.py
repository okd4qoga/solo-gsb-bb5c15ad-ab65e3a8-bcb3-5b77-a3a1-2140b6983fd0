"""Import, rule evaluation and timeline services built on the engine + db."""

from __future__ import annotations

import json
import time
from collections import defaultdict

from . import db
from .engine import (
    Condition,
    Event,
    Interval,
    MaintenanceWindow,
    cluster_intervals,
    diff_faults,
    evaluate_rule,
    to_epoch,
    to_number,
)

DEFAULT_DEVICE_FIELD = "device"
DEFAULT_METRIC_FIELD = "metric"
DEFAULT_TS_FIELD = "ts"
DEFAULT_VALUE_FIELD = "value"


def parse_jsonl(text: str) -> list[dict]:
    """Parse JSONL; a top-level JSON array is also accepted."""
    text = text.strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON body must be an array of event objects")
        return [r for r in data if isinstance(r, dict)]

    rows: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"line {lineno}: event must be a JSON object")
        rows.append(obj)
    return rows


def _extract_metrics(
    row: dict, device_field: str, metric_field: str, ts_field: str, value_field: str
) -> list[tuple[str, str, object]]:
    """Turn one raw row into (device, metric, raw_value) triples.

    Two shapes are supported:
    1. flat  - {"device": "d", "metric": "m", "ts": ..., "value": 3.1}
    2. nested - {"device": "d", "ts": ..., "value": {"temp": 1, "vib": 0.2}}
       (one event per key of the value object; the metric column is ignored)
    """
    device = row.get(device_field)
    if device is None:
        raise ValueError("missing device")
    raw_ts = row.get(ts_field)
    if raw_ts is None:
        raise ValueError("missing timestamp")

    value = row.get(value_field) if value_field in row else row.get(DEFAULT_VALUE_FIELD)
    if isinstance(value, dict):
        if not value:
            raise ValueError("empty nested metric object")
        return [(str(device), str(sub_metric), sub_value) for sub_metric, sub_value in value.items()]

    metric = row.get(metric_field)
    if metric is None:
        raise ValueError("missing metric")
    return [(str(device), str(metric), value)]


def import_events(
    text: str,
    filename: str | None = None,
    device_field: str = DEFAULT_DEVICE_FIELD,
    metric_field: str = DEFAULT_METRIC_FIELD,
    ts_field: str = DEFAULT_TS_FIELD,
    value_field: str = DEFAULT_VALUE_FIELD,
) -> dict:
    rows = parse_jsonl(text)
    now = time.time()

    conn = db.get_conn()
    cur = conn.execute(
        "INSERT INTO imports (filename, imported_at, total_rows) VALUES (?,?,?)",
        (filename, now, len(rows)),
    )
    import_id = cur.lastrowid

    inserted = 0
    overwritten = 0
    skipped: list[dict] = []
    with db.transaction() as tx:
        for index, row in enumerate(rows):
            try:
                triples = _extract_metrics(
                    row, device_field, metric_field, ts_field, value_field
                )
                ts = to_epoch(row[ts_field])
            except (ValueError, KeyError, TypeError) as exc:
                skipped.append({"line": index + 1, "reason": str(exc)})
                continue
            for device, metric, raw_value in triples:
                num = to_number(raw_value)
                if num is None:
                    skipped.append(
                        {"line": index + 1, "reason": f"non-numeric value for {device}/{metric}"}
                    )
                    continue
                is_new = db.upsert_event(
                    tx, device, metric, ts, num, json.dumps(row, ensure_ascii=False), import_id
                )
                if is_new:
                    inserted += 1
                else:
                    overwritten += 1
        tx.execute(
            "UPDATE imports SET inserted_rows = ?, skipped_rows = ? WHERE id = ?",
            (inserted, len(skipped), import_id),
        )

    return {
        "import_id": import_id,
        "total_rows": len(rows),
        "inserted": inserted,
        "overwritten": overwritten,
        "skipped": len(skipped),
        "skipped_details": skipped[:100],
    }


def _load_events_index() -> tuple[dict[tuple[str, str], list[Event]], float | None]:
    events_by_channel: dict[tuple[str, str], list[Event]] = defaultdict(list)
    horizon: float | None = None
    for row in db.query("SELECT id, device, metric, ts, value FROM events ORDER BY ts, id"):
        if row["value"] is None:
            continue
        ev = Event(
            id=row["id"],
            device=row["device"],
            metric=row["metric"],
            ts=row["ts"],
            value=row["value"],
        )
        events_by_channel[(ev.device, ev.metric)].append(ev)
        horizon = ev.ts if horizon is None else max(horizon, ev.ts)
    return events_by_channel, horizon


def _load_maintenance_windows() -> list[MaintenanceWindow]:
    """All stored planned-maintenance windows as engine inputs."""
    return [
        MaintenanceWindow(
            id=w["id"],
            device=w["device"],
            start=w["start"],
            end=w["end"],
            name=w["name"],
        )
        for w in db.fetch_maintenance_windows()
    ]


def evaluate_all_rules() -> dict:
    """Re-evaluate every rule and replace its stored fault intervals.

    Planned-maintenance windows are merged per device and subtracted from the
    truth segments before the duration check, so a maintenance interruption
    resets the sustained duration and re-arms confirmation after the window.
    The suppressed pieces are stored alongside for the timeline/export.
    """
    rules = db.fetch_rules()
    events_by_channel, horizon = _load_events_index()
    if horizon is None:
        horizon = 0.0
    maintenance = _load_maintenance_windows()

    total = 0
    total_suppressed = 0
    per_rule: list[dict] = []
    with db.transaction() as tx:
        for rule in rules:
            conditions = [Condition.from_dict(c) for c in rule["conditions"]]
            result = evaluate_rule(
                rule_id=rule["id"],
                logic=rule["logic"],
                duration_required=rule["duration"],
                conditions=conditions,
                events_by_channel=events_by_channel,
                horizon=horizon,
                maintenance=maintenance,
            )
            tx.execute("DELETE FROM fault_intervals WHERE rule_id = ?", (rule["id"],))
            tx.execute("DELETE FROM suppressed_segments WHERE rule_id = ?", (rule["id"],))
            for f in result.faults:
                tx.execute(
                    """
                    INSERT INTO fault_intervals
                        (rule_id, start_ts, end_ts, duration_required, confirmed_at, evidence_json)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        f.rule_id,
                        f.start,
                        f.end,
                        f.duration_required,
                        f.confirmed_at,
                        json.dumps([e.to_dict() for e in f.evidence], ensure_ascii=False),
                    ),
                )
            for s in result.suppressed:
                tx.execute(
                    """
                    INSERT INTO suppressed_segments
                        (rule_id, device, metric, start_ts, end_ts, maintenance_ids)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        rule["id"],
                        s.device,
                        s.metric,
                        s.interval.start,
                        s.interval.end,
                        json.dumps(s.maintenance_ids),
                    ),
                )
            total += len(result.faults)
            total_suppressed += len(result.suppressed)
            per_rule.append(
                {
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "faults": len(result.faults),
                    "suppressed": len(result.suppressed),
                }
            )

    return {
        "rules_evaluated": len(rules),
        "fault_intervals": total,
        "suppressed_segments": total_suppressed,
        "maintenance_windows": len(maintenance),
        "per_rule": per_rule,
    }


def _fault_outline(fault) -> dict:
    """Compact fault payload for the impact preview (evidence is irrelevant)."""
    return {
        "rule_id": fault.rule_id,
        "start": fault.start,
        "end": fault.end,
        "duration_required": fault.duration_required,
        "confirmed_at": fault.confirmed_at,
    }


def preview_rule(rule_id: int | None, draft: dict) -> dict:
    """Evaluate a not-yet-saved rule draft against the live data, read-only.

    The baseline is the rule *as currently stored* (or nothing for a new
    rule), the candidate is the unsaved draft.  Both are evaluated right now
    with the current events and the currently stored planned-maintenance
    windows — exactly what a save + re-evaluation would see — but nothing is
    written to the rules or fault tables.  The two confirmed fault sets are
    compared as half-open interval coverings (see :func:`diff_faults`).
    """
    events_by_channel, horizon = _load_events_index()
    if horizon is None:
        horizon = 0.0
    maintenance = _load_maintenance_windows()

    baseline_rule = db.fetch_rule(rule_id) if rule_id is not None else None
    if rule_id is not None and baseline_rule is None:
        raise KeyError(f"rule {rule_id} not found")

    def run(rule: dict, rid: int):
        return evaluate_rule(
            rule_id=rid,
            logic=rule["logic"],
            duration_required=rule["duration"],
            conditions=[Condition.from_dict(c) for c in rule["conditions"]],
            events_by_channel=events_by_channel,
            horizon=horizon,
            maintenance=maintenance,
        )

    baseline_result = (
        run(baseline_rule, baseline_rule["id"]) if baseline_rule is not None else None
    )
    draft_result = run(draft, rule_id if rule_id is not None else -1)

    baseline_faults = baseline_result.faults if baseline_result is not None else []
    segments = diff_faults(baseline_faults, draft_result.faults)
    counts = {"added": 0, "removed": 0, "overlap_changed": 0, "overlap_unchanged": 0}
    for seg in segments:
        counts[seg.kind] += 1

    return {
        "rule_id": rule_id,
        "mode": "edit" if baseline_rule is not None else "create",
        "baseline": {
            "name": baseline_rule["name"] if baseline_rule is not None else None,
            "faults": [_fault_outline(f) for f in baseline_faults],
        },
        "draft": {
            "name": draft["name"],
            "faults": [_fault_outline(f) for f in draft_result.faults],
        },
        "maintenance_windows": len(maintenance),
        "diff": {
            "segments": [seg.to_dict() for seg in segments],
            "counts": counts,
            "changed": counts["added"] + counts["removed"] + counts["overlap_changed"] > 0,
        },
    }


def list_channels() -> list[dict]:
    rows = db.query(
        """
        SELECT device, metric, COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts,
               MIN(value) AS min_value, MAX(value) AS max_value
        FROM events WHERE value IS NOT NULL
        GROUP BY device, metric ORDER BY device, metric
        """
    )
    return [dict(r) for r in rows]


def list_events(
    start: float | None = None,
    end: float | None = None,
    device: str | None = None,
    metric: str | None = None,
    limit: int = 2000,
    offset: int = 0,
) -> tuple[list[dict], int]:
    where = []
    params: list = []
    # Point events: inclusive range on both ends.
    if start is not None:
        where.append("ts >= ?")
        params.append(start)
    if end is not None:
        where.append("ts <= ?")
        params.append(end)
    if device:
        where.append("device = ?")
        params.append(device)
    if metric:
        where.append("metric = ?")
        params.append(metric)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    total = db.query_one(f"SELECT COUNT(*) AS c FROM events{clause}", params)["c"]
    rows = db.query(
        f"SELECT id, device, metric, ts, value, import_id FROM events{clause} ORDER BY ts, id LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )
    return [dict(r) for r in rows], total


def query_faults(
    start: float | None = None, end: float | None = None
) -> list[dict]:
    where = []
    params: list = []
    if start is not None:
        where.append("end_ts > ?")  # half-open overlap: [start,end) ∩ range
        params.append(start)
    if end is not None:
        where.append("start_ts < ?")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.query(
        f"""
        SELECT f.*, r.name AS rule_name, r.logic
        FROM fault_intervals f JOIN rules r ON r.id = f.rule_id
        {clause} ORDER BY f.start_ts, f.rule_id
        """,
        params,
    )
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "rule_id": row["rule_id"],
                "rule_name": row["rule_name"],
                "logic": row["logic"],
                "start": row["start_ts"],
                "end": row["end_ts"],
                "duration_required": row["duration_required"],
                "confirmed_at": row["confirmed_at"],
                "evidence": json.loads(row["evidence_json"]),
            }
        )
    return out


def _overlaps_window(start: float, end: float, win_start: float | None, win_end: float | None) -> bool:
    """Half-open overlap with the filter window [win_start, win_end)."""
    if win_start is not None and end <= win_start:
        return False
    if win_end is not None and start >= win_end:
        return False
    return True


def _fault_member(fault: dict, in_window: bool) -> dict:
    """One member fault as embedded inside a fault event.

    The payload is a copy of the original stored fault (id, interval and
    trigger evidence untouched) plus the window-membership flag; the event
    view never rewrites fault results.
    """
    return {
        "fault_id": fault["id"],
        "rule_id": fault["rule_id"],
        "rule_name": fault["rule_name"],
        "logic": fault["logic"],
        "start": fault["start"],
        "end": fault["end"],
        "duration_required": fault["duration_required"],
        "confirmed_at": fault["confirmed_at"],
        "in_window": in_window,
        "evidence": fault["evidence"],
    }


def build_fault_events(
    faults: list[dict], start: float | None = None, end: float | None = None
) -> list[dict]:
    """Group faults from different rules into fault *events*.

    Two faults (typically produced by different rules) belong to the same
    event when their ``[start,end)`` intervals overlap or touch end-to-end;
    the relation is transitive, so chaining links are merged as well.  An
    event spans ``[min member start, max member end)``.

    Components are first computed over *all* faults (so chains crossing the
    filter window stay intact), then only events overlapping the
    half-open filter window are returned.  Members outside the window are
    still listed, flagged ``in_window: false``.
    """
    groups = cluster_intervals([Interval(f["start"], f["end"]) for f in faults])
    events: list[dict] = []
    for members in groups:
        member_faults = [faults[i] for i in members]
        ev_start = min(f["start"] for f in member_faults)
        ev_end = max(f["end"] for f in member_faults)
        if not _overlaps_window(ev_start, ev_end, start, end):
            continue
        member_payload = [
            _fault_member(f, _overlaps_window(f["start"], f["end"], start, end))
            for f in sorted(member_faults, key=lambda f: (f["start"], f["end"], f["id"]))
        ]
        rule_ids = sorted({f["rule_id"] for f in member_faults})
        events.append(
            {
                "start": ev_start,
                "end": ev_end,
                "rule_count": len(rule_ids),
                "fault_count": len(member_payload),
                "member_count_in_window": sum(1 for m in member_payload if m["in_window"]),
                "rule_ids": rule_ids,
                "rule_names": sorted({f["rule_name"] for f in member_faults}),
                "members": member_payload,
            }
        )
    events.sort(key=lambda e: (e["start"], e["end"]))
    # Sequential display id, stable within the queried time range.
    for index, event in enumerate(events, start=1):
        event["id"] = f"E{index}"
    return events


def query_fault_events(
    start: float | None = None, end: float | None = None
) -> list[dict]:
    """Fault events for the filter window; clustering sees every stored fault."""
    return build_fault_events(query_faults(None, None), start, end)


def query_suppressed(
    start: float | None = None, end: float | None = None
) -> list[dict]:
    """Suppressed segments (truth lost to maintenance), half-open filtered.

    Each segment is linked to its rule and to the maintenance windows that
    suppressed it (ids resolved to names for display/export).
    """
    where = []
    params: list = []
    if start is not None:
        where.append("s.end_ts > ?")
        params.append(start)
    if end is not None:
        where.append("s.start_ts < ?")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.query(
        f"""
        SELECT s.*, r.name AS rule_name
        FROM suppressed_segments s JOIN rules r ON r.id = s.rule_id
        {clause} ORDER BY s.start_ts, s.rule_id, s.id
        """,
        params,
    )
    windows = {w["id"]: w for w in db.fetch_maintenance_windows()}
    out = []
    for row in rows:
        linked = []
        for wid in json.loads(row["maintenance_ids"]):
            w = windows.get(wid)
            linked.append(
                {"id": wid, "name": w["name"], "device": w["device"]}
                if w
                else {"id": wid, "name": None, "device": row["device"]}
            )
        out.append(
            {
                "id": row["id"],
                "rule_id": row["rule_id"],
                "rule_name": row["rule_name"],
                "device": row["device"],
                "metric": row["metric"],
                "start": row["start_ts"],
                "end": row["end_ts"],
                "maintenance_ids": json.loads(row["maintenance_ids"]),
                "maintenance_windows": linked,
            }
        )
    return out


def query_annotations(start: float | None = None, end: float | None = None) -> list[dict]:
    where = []
    params: list = []
    if start is not None:
        where.append("end_ts > ?")
        params.append(start)
    if end is not None:
        where.append("start_ts < ?")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.query(
        f"SELECT * FROM annotations{clause} ORDER BY start_ts, id", params
    )
    return [
        {
            "id": r["id"],
            "start": r["start_ts"],
            "end": r["end_ts"],
            "label": r["label"],
            "color": r["color"],
            "note": r["note"],
        }
        for r in rows
    ]


def downsample(values: list[tuple[float, float, int]], max_points: int) -> list[tuple[float, float, int]]:
    """Stride downsample; first/last points are always kept.

    values: (ts, value, event_id) sorted by ts.  A stride (rather than bucket
    averaging) preserves the step-plot shape of held signals.
    """
    if len(values) <= max_points or max_points < 2:
        return values
    step = len(values) / max_points
    picked = [values[0]]
    index = step
    while index < len(values) - 1:
        picked.append(values[int(index)])
        index += step
    picked.append(values[-1])
    return picked


def timeline_data(start: float | None, end: float | None, max_points: int = 400) -> dict:
    channels = list_channels()
    channels_in_range = []
    for ch in channels:
        if start is not None and ch["max_ts"] < start:
            continue
        if end is not None and ch["min_ts"] > end:
            continue
        channels_in_range.append(ch)

    channel_payload = []
    for ch in channels_in_range:
        sql = (
            "SELECT ts, value, id FROM events WHERE device = ? AND metric = ? AND value IS NOT NULL"
        )
        params: list = [ch["device"], ch["metric"]]
        if start is not None:
            sql += " AND ts >= ?"
            params.append(start)
        if end is not None:
            sql += " AND ts <= ?"
            params.append(end)
        sql += " ORDER BY ts, id"
        rows = db.query(sql, params)
        values = [(r["ts"], r["value"], r["id"]) for r in rows]
        total = len(values)
        values = downsample(values, max_points)
        channel_payload.append(
            {
                "device": ch["device"],
                "metric": ch["metric"],
                "count": ch["n"],
                "shown": total,
                "min_ts": ch["min_ts"],
                "max_ts": ch["max_ts"],
                "min_value": ch["min_value"],
                "max_value": ch["max_value"],
                "values": [{"ts": t, "v": v, "id": i} for t, v, i in values],
            }
        )

    bounds = db.query_one("SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM events")
    return {
        "channels": channel_payload,
        "faults": query_faults(start, end),
        "annotations": query_annotations(start, end),
        "maintenance": db.fetch_maintenance_windows(start, end),
        "suppressed": query_suppressed(start, end),
        "data_start": bounds["lo"],
        "data_end": bounds["hi"],
    }
