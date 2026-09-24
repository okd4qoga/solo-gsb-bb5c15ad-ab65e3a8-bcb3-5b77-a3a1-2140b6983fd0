"""Tests for per-channel sample validity (TTL).

A configured TTL (positive seconds) clips every held piece to
``[sample.ts, min(next_sample.ts, sample.ts + ttl, horizon))``; expired gaps
satisfy no condition, AND/OR combine only the valid pieces, the sustained
duration re-accumulates after a truth break, and fault intervals,
confirmation instants and trigger evidence never cross an invalid gap.
Unconfigured channels keep the legacy hold-until-next-sample semantics.

Two layers are covered:

* engine — ``channel_held_intervals(..., ttl=)`` and ``evaluate_rule`` with
  ``sample_ttls`` (including TTL + maintenance interaction);
* services — the scratch-DB round trips: set/clear TTL, re-evaluation after
  TTL changes and after late-sample back-fill, and read-only draft preview
  against the current TTL configuration.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import (
    Condition,
    Event,
    Interval,
    MaintenanceWindow,
    channel_held_intervals,
    evaluate_rule,
    to_epoch,
)


def ev(eid, ts, value, device="d", metric="m"):
    return Event(id=eid, device=device, metric=metric, ts=ts, value=value)


def minute_series(eid_base, predicate_fn, device="d", metric="m"):
    """One sample per minute for minutes 0..30, value from predicate_fn(minute)."""
    return [
        ev(eid_base + m, m * 60, predicate_fn(m), device, metric)
        for m in range(31)
    ]


def ivs(faults):
    return [(f.start, f.end) for f in faults]


def mw(wid, device, start_min, end_min, name="检修"):
    return MaintenanceWindow(
        id=wid, device=device, start=start_min * 60, end=end_min * 60, name=name
    )


# --------------------------------------------------------------------------- #
# engine: channel_held_intervals with TTL
# --------------------------------------------------------------------------- #


def test_ttl_clips_last_sample_before_horizon():
    # samples at 0 and 60; TTL 30s: held [0,30), [60,90) (capped by ttl, not horizon)
    events = [ev(1, 0, 90), ev(2, 60, 90)]
    cond = Condition("d", "m", ">", 50)
    held, witnesses = channel_held_intervals(events, cond, horizon=300, ttl=30)
    assert held == [Interval(0, 30), Interval(60, 90)]
    assert [w.id for w in witnesses] == [1, 2]


def test_ttl_short_of_next_sample_leaves_real_gap():
    # samples every 60s, TTL 30s: each held piece is [m*60, m*60+30);
    # touching-style union must NOT bridge the 30s invalid gaps.
    events = minute_series(100, lambda m: 90)
    cond = Condition("d", "m", ">", 50)
    held, _ = channel_held_intervals(events, cond, horizon=30 * 60, ttl=30)
    assert held[0] == Interval(0, 30)
    assert held[1] == Interval(60, 90)
    # minutes 0..29 yield 30 pieces; the minute-30 sample's clip
    # [1800, min(1800+30, horizon=1800)) is empty and dropped
    assert len(held) == 30
    # real 30s gaps between the pieces
    for a, b in zip(held, held[1:]):
        assert b.start - a.end == 30


def test_ttl_equal_to_sample_cadence_matches_legacy_hold():
    # TTL 60s with one sample per minute: pieces touch at the next sample,
    # exactly reproducing the legacy hold-until-next-sample semantics.
    events = minute_series(100, lambda m: 90)
    cond = Condition("d", "m", ">", 50)
    held_ttl, _ = channel_held_intervals(events, cond, horizon=30 * 60, ttl=60)
    held_legacy, _ = channel_held_intervals(events, cond, horizon=30 * 60)
    assert held_ttl == held_legacy


def test_ttl_longer_than_cadence_is_clipped_by_next_sample():
    events = [ev(1, 0, 90), ev(2, 60, 10)]
    cond = Condition("d", "m", ">", 50)
    held, _ = channel_held_intervals(events, cond, horizon=300, ttl=120)
    # the next sample (60, low value) ends the hold before ts+ttl=120
    assert held == [Interval(0, 60)]


def test_ttl_caps_hold_at_horizon():
    events = [ev(1, 250, 90)]
    cond = Condition("d", "m", ">", 50)
    held, _ = channel_held_intervals(events, cond, horizon=300, ttl=1000)
    assert held == [Interval(250, 300)]


def test_ttl_false_value_no_held_piece():
    events = [ev(1, 0, 10)]
    cond = Condition("d", "m", ">", 50)
    held, witnesses = channel_held_intervals(events, cond, horizon=300, ttl=30)
    assert held == [] and witnesses == []


# --------------------------------------------------------------------------- #
# engine: evaluate_rule with sample_ttls
# --------------------------------------------------------------------------- #


def test_ttl_gaps_filter_fault_by_duration_and_reset():
    # high every minute for minutes 6..16; TTL 30s makes every true piece
    # only 30s long, so a 5-minute rule has no fault at all.
    events = minute_series(100, lambda m: 90 if 6 <= m <= 16 else 10)
    cond = Condition("d", "m", ">=", 80)
    result = evaluate_rule(
        1, "AND", 300, [cond], {("d", "m"): events},
        horizon=30 * 60, sample_ttls={("d", "m"): 30},
    )
    assert result.faults == []


def test_ttl_continuous_pieces_still_confirm_when_duration_fits():
    # samples every minute, TTL exactly 60s: pieces touch at each next sample,
    # so the [6,17) truth stays one continuous segment as in the legacy case.
    events = minute_series(100, lambda m: 90 if 6 <= m <= 16 else 10)
    cond = Condition("d", "m", ">=", 80)
    result = evaluate_rule(
        1, "AND", 300, [cond], {("d", "m"): events},
        horizon=30 * 60, sample_ttls={("d", "m"): 60},
    )
    assert ivs(result.faults) == [(6 * 60, 17 * 60)]
    assert result.faults[0].confirmed_at == 6 * 60 + 300


def test_ttl_truth_break_resets_sustained_duration():
    # minute cadence except a two-minute hole between m=8 and m=10 samples;
    # TTL 60s clips the m=8 piece to [8,9) instead of [8,10). The continuous
    # runs [6,9) and [10,17) must be separate faults, each confirming from
    # its own start; with a 4-minute rule the first run disappears.
    events = [
        ev(100 + m, m * 60, 90 if 6 <= m <= 16 else 10)
        for m in range(31) if m != 9
    ]
    cond = Condition("d", "m", ">=", 80)
    result = evaluate_rule(
        1, "AND", 4 * 60, [cond], {("d", "m"): events},
        horizon=30 * 60, sample_ttls={("d", "m"): 60},
    )
    assert ivs(result.faults) == [(10 * 60, 17 * 60)]
    assert result.faults[0].confirmed_at == 10 * 60 + 4 * 60


def test_ttl_clips_last_sample_hold_after_final_sample():
    # The channel's last sample (at minute 16) is high and nothing follows it.
    # Legacy hold runs [16, horizon=30); a TTL of 60s cuts it at 17:00, so a
    # 5-minute duration rule has no fault under TTL.
    events = [ev(100 + m, m * 60, 10) for m in range(17)]
    events[-1] = ev(100 + 16, 16 * 60, 90)
    cond = Condition("d", "m", ">=", 80)
    legacy = evaluate_rule(
        1, "AND", 300, [cond], {("d", "m"): events}, horizon=30 * 60
    )
    assert ivs(legacy.faults) == [(16 * 60, 30 * 60)]
    with_ttl = evaluate_rule(
        1, "AND", 300, [cond], {("d", "m"): events},
        horizon=30 * 60, sample_ttls={("d", "m"): 60},
    )
    assert with_ttl.faults == []
    # with a zero duration the clipped piece [16,17) remains visible
    zero_dur = evaluate_rule(
        1, "AND", 0, [cond], {("d", "m"): events},
        horizon=30 * 60, sample_ttls={("d", "m"): 60},
    )
    assert ivs(zero_dur.faults) == [(16 * 60, 17 * 60)]


def test_ttl_evidence_held_intervals_and_witness_samples_never_cross_gap():
    # hole at minute 9 + TTL 60s: the surviving [10,17) fault's evidence must
    # start exactly at 10*60 and reference no sample from before the gap.
    events = [
        ev(100 + m, m * 60, 90 if 6 <= m <= 16 else 10)
        for m in range(31) if m != 9
    ]
    cond = Condition("d", "m", ">=", 80)
    result = evaluate_rule(
        1, "AND", 2 * 60, [cond], {("d", "m"): events},
        horizon=30 * 60, sample_ttls={("d", "m"): 60},
    )
    starts = [s for s, _ in ivs(result.faults)]
    assert (10 * 60, 17 * 60) in ivs(result.faults)
    tail = next(f for f in result.faults if f.start == 10 * 60)
    evidence = tail.evidence[0]
    assert all(iv.start >= 10 * 60 for iv in evidence.held_intervals)
    assert min(w.ts for w in evidence.events) == 10 * 60
    # the earlier [6,9) run (3 minutes) also qualifies at a 2-minute duration,
    # but its evidence is clipped at the expiry instant 9*60
    head = next(f for f in result.faults if f.start == 6 * 60)
    assert head.end == 9 * 60
    assert head.evidence[0].held_intervals == [Interval(6 * 60, 9 * 60)]
    assert [w.ts for w in head.evidence[0].events] == [6 * 60, 7 * 60, 8 * 60]


def test_ttl_and_does_not_bridge_invalid_gap():
    # temp high throughout; vib has a missing minute 9 with TTL 60s.
    # AND truth must break at the vib expiry: [6,9) and [10,17).
    temp = minute_series(100, lambda m: 90, device="d", metric="temp")
    vib = [
        ev(200 + m, m * 60, 3.6, device="d", metric="vib")
        for m in range(31) if m != 9
    ]
    result = evaluate_rule(
        2, "AND", 0,
        [Condition("d", "temp", ">=", 80), Condition("d", "vib", ">", 3)],
        {("d", "temp"): temp, ("d", "vib"): vib},
        horizon=30 * 60,
        sample_ttls={("d", "temp"): 60, ("d", "vib"): 60},
    )
    assert ivs(result.faults) == [(0, 9 * 60), (10 * 60, 30 * 60)]


def test_ttl_and_without_config_on_one_channel_keeps_legacy_hold():
    # only vib has a TTL; temp holds to the next sample as before
    temp = minute_series(100, lambda m: 90, device="d", metric="temp")
    vib = [
        ev(200 + m, m * 60, 3.6 if m != 9 else 1.0, device="d", metric="vib")
        for m in range(31)
    ]
    result = evaluate_rule(
        2, "AND", 0,
        [Condition("d", "temp", ">=", 80), Condition("d", "vib", ">", 3)],
        {("d", "temp"): temp, ("d", "vib"): vib},
        horizon=30 * 60,
        sample_ttls={("d", "vib"): 60},
    )
    # vib false at minute 9 already breaks [9,10); with TTL 60 the pieces
    # [m,m+1) touch at the next samples — same as the plain value result
    assert ivs(result.faults) == [
        (0, 9 * 60), (10 * 60, 30 * 60),
    ]


def test_ttl_or_combines_valid_pieces_without_bridging_gaps():
    # vib high [3,9) with a missing minute 6 (TTL 60 clips piece at 7);
    # press low on [6,9) (no hole). OR covers [3,7) via vib and [6,9) via
    # press -> [3,9), but after vib expires at 7 and press ends at 9 there
    # is no bridge to later truth. Separate press dip [10,14) stays separate.
    vib = [
        ev(300 + m, m * 60, 6.0 if (3 <= m <= 8 and m != 6) else 2.0,
           device="d", metric="vib")
        for m in range(31)
    ]
    press = minute_series(
        400, lambda m: 26.0 if (6 <= m <= 8 or 10 <= m <= 13) else 52.0,
        device="d", metric="press",
    )
    result = evaluate_rule(
        3, "OR", 0,
        [Condition("d", "vib", ">", 5), Condition("d", "press", "<", 30)],
        {("d", "vib"): vib, ("d", "press"): press},
        horizon=30 * 60, sample_ttls={("d", "vib"): 60},
    )
    assert ivs(result.faults) == [(3 * 60, 9 * 60), (10 * 60, 14 * 60)]


def test_ttl_with_maintenance_still_subtracts_and_reconfirms():
    # TTL 60s (continuous at the 1-minute cadence) + maintenance [10,12):
    # the maintenance semantics must be unchanged on top of TTL clipping.
    events = minute_series(100, lambda m: 90 if 6 <= m <= 16 else 10)
    cond = Condition("d", "m", ">=", 80)
    result = evaluate_rule(
        1, "AND", 300, [cond], {("d", "m"): events},
        horizon=30 * 60, maintenance=[mw(1, "d", 10, 12)],
        sample_ttls={("d", "m"): 60},
    )
    assert ivs(result.faults) == [(12 * 60, 17 * 60)]
    assert result.faults[0].confirmed_at == 12 * 60 + 300
    assert [(s.interval.start, s.interval.end) for s in result.suppressed] == [
        (10 * 60, 12 * 60)
    ]


def test_ttl_gap_plus_maintenance_cut_produce_separate_faults():
    # missing minute 9 (TTL expiry at 9) and maintenance [11,13):
    # truth [6,9), [10,11) (cut by the window), [13,17) — three faults
    # at a 60-second duration, each re-accumulating on its own.
    events = [
        ev(100 + m, m * 60, 90 if 6 <= m <= 16 else 10)
        for m in range(31) if m != 9
    ]
    cond = Condition("d", "m", ">=", 80)
    result = evaluate_rule(
        1, "AND", 60, [cond], {("d", "m"): events},
        horizon=30 * 60, maintenance=[mw(1, "d", 11, 13)],
        sample_ttls={("d", "m"): 60},
    )
    assert ivs(result.faults) == [
        (6 * 60, 9 * 60),
        (10 * 60, 11 * 60),
        (13 * 60, 17 * 60),
    ]
    assert [f.confirmed_at for f in result.faults] == [
        7 * 60, 11 * 60, 14 * 60,
    ]


def test_unconfigured_channel_keeps_legacy_semantics_with_ttl_map_present():
    events = minute_series(100, lambda m: 90 if 6 <= m <= 16 else 10)
    cond = Condition("d", "m", ">=", 80)
    # map present but no entry for this channel
    result = evaluate_rule(
        1, "AND", 300, [cond], {("d", "m"): events},
        horizon=30 * 60, sample_ttls={("other", "x"): 5},
    )
    assert ivs(result.faults) == [(6 * 60, 17 * 60)]


# --------------------------------------------------------------------------- #
# services: scratch-DB round trips
# --------------------------------------------------------------------------- #


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def _import_minute_series(true_minutes=range(6, 17), gap_minutes=()):
    rows = [
        {
            "device": "d",
            "metric": "m",
            "ts": f"2026-09-23T00:{m:02d}:00Z",
            "value": 90 if m in true_minutes else 10,
        }
        for m in range(31)
        if m not in gap_minutes
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows))


def _create_stored_rule(duration=300):
    rule_id = db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("TTL 规则", "AND", duration, 0.0),
    ).lastrowid
    db.execute(
        "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
        " VALUES (?,?,?,?,?,?)",
        (rule_id, "d", "m", ">=", 80, 0),
    )
    return rule_id


def at(minute: int) -> float:
    return to_epoch(f"2026-09-23T00:{minute:02d}:00Z")


def test_services_set_and_fetch_ttl_validates_positive_seconds(scratch_db):
    saved = services.set_sample_ttl("d", "m", 90)
    assert saved["ttl_seconds"] == 90.0
    assert saved["device"] == "d" and saved["metric"] == "m"
    listed = services.list_sample_ttls()
    assert len(listed) == 1
    assert listed[0]["device"] == "d"
    assert listed[0]["metric"] == "m"
    assert listed[0]["ttl_seconds"] == 90.0
    # upsert replaces the value and keeps created_at
    created_first = saved["created_at"]
    updated = services.set_sample_ttl("d", "m", 45)
    assert updated["ttl_seconds"] == 45.0
    assert updated["created_at"] == created_first
    assert len(services.list_sample_ttls()) == 1

    for bad in (0, -1, "abc", float("inf")):
        with pytest.raises(ValueError):
            services.set_sample_ttl("d", "x", bad)


def test_services_set_ttl_recomputes_stored_faults(scratch_db):
    _import_minute_series()
    _create_stored_rule(duration=300)
    services.evaluate_all_rules()
    assert [(f["start"], f["end"]) for f in services.query_faults(None, None)] == [
        (at(6), at(17))
    ]

    # TTL 30s turns each held minute into a 30s piece: no 5-minute fault
    services.set_sample_ttl("d", "m", 30)
    assert services.query_faults(None, None) == []

    # TTL 60s reproduces the touching-piece legacy result
    services.set_sample_ttl("d", "m", 60)
    assert [(f["start"], f["end"]) for f in services.query_faults(None, None)] == [
        (at(6), at(17))
    ]


def test_services_clear_ttl_restores_legacy_hold(scratch_db):
    _import_minute_series()
    _create_stored_rule(duration=300)
    services.set_sample_ttl("d", "m", 30)
    assert services.query_faults(None, None) == []

    cleared = services.clear_sample_ttl("d", "m")
    assert cleared["deleted"] is True
    assert services.list_sample_ttls() == []
    assert [(f["start"], f["end"]) for f in services.query_faults(None, None)] == [
        (at(6), at(17))
    ]
    with pytest.raises(KeyError):
        services.clear_sample_ttl("d", "m")


def test_services_backfill_old_sample_recomputes_existing_faults(scratch_db):
    # import the series WITHOUT minute 9; TTL 60s clips the m=8 hold at 9:00.
    _import_minute_series(gap_minutes=(9,))
    _create_stored_rule(duration=240)
    services.set_sample_ttl("d", "m", 60)
    faults = [(f["start"], f["end"]) for f in services.query_faults(None, None)]
    # [6,9) is only 3 min (<240s); [10,17) is 7 min -> one fault
    assert faults == [(at(10), at(17))]

    # late-arriving sample at minute 9 back-fills the gap; import must
    # recompute stored faults, restoring the continuous [6,17) truth
    # (pieces [m,m+1) touch at each newly present sample, TTL never expires)
    backfill = {"device": "d", "metric": "m",
                "ts": "2026-09-23T00:09:00Z", "value": 90}
    result = services.import_events(json.dumps(backfill))
    assert "evaluation" in result and result["evaluation"]["rules_evaluated"] == 1
    faults = [(f["start"], f["end"]) for f in services.query_faults(None, None)]
    assert faults == [(at(6), at(17))]


def test_services_import_without_rules_does_not_evaluate(scratch_db):
    _import_minute_series()
    result = services.import_events(
        json.dumps({"device": "d", "metric": "m",
                    "ts": "2026-09-23T00:05:00Z", "value": 90})
    )
    assert "evaluation" not in result


def test_services_preview_uses_current_ttl_without_writing(scratch_db):
    _import_minute_series(gap_minutes=(9,))
    rule_id = _create_stored_rule(duration=240)
    services.evaluate_all_rules()
    services.set_sample_ttl("d", "m", 60)

    # baseline with current TTL: only [10,17) survives; the draft (duration 60)
    # would additionally confirm the [6,9) piece — shown as added.
    result = services.preview_rule(
        rule_id,
        {
            "name": "TTL 规则",
            "logic": "AND",
            "duration": 60,
            "conditions": [
                {"device": "d", "metric": "m", "op": ">=", "threshold": 80}
            ],
        },
    )
    assert result["sample_ttls"] == 1
    assert [(f["start"], f["end"]) for f in result["baseline"]["faults"]] == [
        (at(10), at(17))
    ]
    assert sorted((f["start"], f["end"]) for f in result["draft"]["faults"]) == [
        (at(6), at(9)),
        (at(10), at(17)),
    ]
    added = [(s["start"], s["end"]) for s in result["diff"]["segments"] if s["kind"] == "added"]
    assert added == [(at(6), at(9))]
    # preview wrote nothing into the TTL or fault tables
    assert len(services.list_sample_ttls()) == 1
    assert [(f["start"], f["end"]) for f in services.query_faults(None, None)] == [
        (at(10), at(17))
    ]


def test_services_stored_evidence_does_not_cross_ttl_gap(scratch_db):
    _import_minute_series(gap_minutes=(9,))
    _create_stored_rule(duration=120)
    services.set_sample_ttl("d", "m", 60)
    faults = services.query_faults(None, None)
    assert [(f["start"], f["end"]) for f in faults] == [
        (at(6), at(9)),
        (at(10), at(17)),
    ]
    for f in faults:
        group = f["evidence"][0]
        for start, end in group["held_intervals"]:
            assert f["start"] - 1e-6 <= start and end <= f["end"] + 1e-6
        for e in group["events"]:
            assert f["start"] <= e["ts"] < f["end"]
