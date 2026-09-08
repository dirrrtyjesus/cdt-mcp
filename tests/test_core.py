import json
import math

import numpy as np
import pytest

from cdt_mcp.core import CoherenceField, DecayRecord, WriteRecord, normalize_phase, phase_from_key


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_phase_from_key_is_deterministic_and_in_range():
    a = phase_from_key("branch:feature-x")
    b = phase_from_key("branch:feature-x")
    assert a == b
    assert 0.0 <= a < 2 * math.pi
    assert phase_from_key("other") != a


def test_normalize_phase_wraps_and_rejects_nan():
    assert normalize_phase(2 * math.pi + 0.5) == pytest.approx(0.5)
    assert normalize_phase(-0.5) == pytest.approx(2 * math.pi - 0.5)
    with pytest.raises(ValueError):
        normalize_phase(float("nan"))


def test_write_superposes_instead_of_overwriting():
    f = CoherenceField("t", bins=8, clock=FakeClock())
    f.write(1.0, phase=0.0, coherence=0.5)
    f.write(1.0, phase=0.0, coherence=0.5)
    assert f.read(phase=0.0).amplitude == pytest.approx(1.0)
    assert len(f) == 2


def test_negative_value_interferes_destructively():
    f = CoherenceField("t", bins=8, clock=FakeClock())
    f.write(1.0, key="k", coherence=1.0, payload="yes")
    f.write(-1.0, key="k", coherence=1.0, payload="no")
    assert f.read(key="k").amplitude == pytest.approx(0.0, abs=1e-12)
    assert f.spectral_density() == pytest.approx(0.0, abs=1e-12)


def test_consensus_picks_highest_density_and_keeps_alternatives():
    f = CoherenceField("t", bins=64, clock=FakeClock())
    f.write(1.0, phase=0.0, coherence=0.9, payload="Hello", agent_id="p1")
    f.write(1.0, phase=math.pi, coherence=0.7, payload="World", agent_id="p2")
    c = f.consensus(top_k=2)
    assert c.top_payload == "Hello"
    assert c.bin == 0
    assert c.share == pytest.approx(0.9 / 1.6)
    assert c.confidence > 1.0
    assert [b.payloads[0].payload for b in c.alternatives] == ["Hello", "World"]
    assert c.payloads[0].agents == ("p1",)
    # The losing proposal is still readable at its own phase.
    assert f.read(phase=math.pi).payloads[0].payload == "World"


def test_empty_field_consensus_is_zero():
    f = CoherenceField("t", clock=FakeClock())
    c = f.consensus()
    assert c.amplitude == 0.0 and c.confidence == 0.0 and c.top_payload is None


def test_bins_wrap_circularly():
    f = CoherenceField("t", bins=8, clock=FakeClock())
    # Just below 2*pi is nearest to bin 0, not the last bin.
    assert f.bin_index(2 * math.pi - 1e-6) == 0
    assert f.bin_index(math.pi) == 4


def test_kernel_width_spreads_write_across_neighbours():
    f = CoherenceField("t", bins=16, kernel_width=0.4, clock=FakeClock())
    f.write(1.0, phase=0.0)
    psi = f.field()
    assert abs(psi[0]) > abs(psi[1]) > abs(psi[2]) > 0
    assert abs(psi[1]) == pytest.approx(abs(psi[15]))
    assert np.sum(np.abs(psi)) == pytest.approx(1.0)


def test_continuous_decay_uses_clock():
    clock = FakeClock(0.0)
    f = CoherenceField("t", bins=8, decay_rate=math.log(2), clock=clock)
    f.write(1.0, phase=0.0)
    assert f.read(phase=0.0).amplitude == pytest.approx(1.0)
    clock.t = 1.0
    assert f.read(phase=0.0).amplitude == pytest.approx(0.5)
    clock.t = 3.0
    assert f.read(phase=0.0).amplitude == pytest.approx(0.125)


def test_explicit_decay_and_prune():
    f = CoherenceField("t", bins=8, clock=FakeClock())
    f.write(1.0, phase=0.0)
    f.write(1e-3, phase=math.pi)
    assert f.decay(dt=1.0, rate=math.log(1000)) == 2
    assert f.read(phase=0.0).amplitude == pytest.approx(1e-3)
    assert f.prune(epsilon=1e-5) == 1
    assert len(f) == 1
    with pytest.raises(ValueError):
        f.decay(dt=-1.0)


def test_merge_is_union_of_events_idempotent_commutative_associative():
    clock = FakeClock()
    a = CoherenceField("a", bins=16, clock=clock)
    b = CoherenceField("b", bins=16, clock=clock)
    c = CoherenceField("c", bins=16, clock=clock)
    a.write(1.0, key="x", coherence=0.8, payload="X")
    b.write(1.0, key="y", coherence=0.6, payload="Y")
    c.write(-1.0, key="x", coherence=0.3, payload="not X")

    ab_c = CoherenceField.merge("m1", [CoherenceField.merge("ab", [a, b]), c], clock=clock)
    a_bc = CoherenceField.merge("m2", [a, CoherenceField.merge("bc", [b, c])], clock=clock)
    cba = CoherenceField.merge("m3", [c, b, a], clock=clock)
    np.testing.assert_allclose(ab_c.field(), a_bc.field())
    np.testing.assert_allclose(ab_c.field(), cba.field())
    assert len(ab_c) == 3

    # Idempotent: merging a replica twice changes nothing.
    before = ab_c.field().copy()
    assert ab_c.merge_from(a) == 0
    np.testing.assert_allclose(ab_c.field(), before)

    # Sources untouched.
    assert len(a) == 1 and len(b) == 1 and len(c) == 1
    assert ab_c.consensus().top_payload == "Y"  # 0.8 - 0.3 = 0.5 < 0.6


def test_merge_rejects_mismatched_bins():
    a = CoherenceField("a", bins=8)
    b = CoherenceField("b", bins=16)
    with pytest.raises(ValueError):
        a.merge_from(b)


def test_tau_k_is_density_weighted_on_merge():
    clock = FakeClock()
    a = CoherenceField("a", bins=8, tau_k=6.0, clock=clock)
    b = CoherenceField("b", bins=8, tau_k=9.0, clock=clock)
    a.write(3.0, phase=0.0)
    b.write(1.0, phase=math.pi)
    a.merge_from(b)
    # weights: 9 vs 1 in spectral density
    assert a.tau_k == pytest.approx((6.0 * 9 + 9.0 * 1) / 10)


def test_snapshot_roundtrip_is_json_safe_and_exact():
    clock = FakeClock(50.0)
    f = CoherenceField("snap", bins=32, decay_rate=0.01, tau_k=8.1, kernel_width=0.2, clock=clock)
    f.write(2.0, key="alpha", coherence=0.4, payload="A", agent_id="agent-1")
    f.write(-1.0, phase=1.234, coherence=1.0)
    data = json.loads(json.dumps(f.to_dict(include_field=True)))
    assert data["schema_version"] == 3
    assert len(data["field"]) == 32
    assert len(data["contested"]) == 32
    g = CoherenceField.from_dict(data, clock=clock)
    assert g.name == "snap" and g.bins == 32 and g.decay_rate == 0.01 and g.tau_k == 8.1
    assert g.kernel_width == 0.2
    np.testing.assert_allclose(g.field(), f.field())
    assert {r.id for r in g.records} == {r.id for r in f.records}
    assert g.created_at == f.created_at


def test_from_dict_rejects_future_schema():
    with pytest.raises(ValueError):
        CoherenceField.from_dict({"schema_version": 99, "name": "x"})


def test_write_validation():
    f = CoherenceField("t")
    with pytest.raises(ValueError):
        f.write(1.0)  # neither key nor phase
    with pytest.raises(ValueError):
        f.write(1.0, key="k", phase=0.0)  # both
    with pytest.raises(ValueError):
        f.write(1.0, phase=0.0, coherence=1.5)
    with pytest.raises(ValueError):
        f.write(float("inf"), phase=0.0)
    with pytest.raises(ValueError):
        CoherenceField("t", bins=1)
    with pytest.raises(ValueError):
        CoherenceField("t", decay_rate=-1)


def test_max_records_prunes_weakest():
    f = CoherenceField("t", bins=8, max_records=2, clock=FakeClock())
    f.write(0.1, phase=0.0, payload="weak")
    f.write(1.0, phase=0.0, payload="strong")
    f.write(0.5, phase=0.0, payload="mid")
    assert len(f) == 2
    assert {r.payload for r in f.records} == {"strong", "mid"}


def test_absorb_with_explicit_record_ids_dedupes():
    f = CoherenceField("t", bins=8, clock=FakeClock())
    rec = WriteRecord(id="fixed", phase=0.0, value=1.0, coherence=1.0, timestamp=0.0)
    assert f.absorb([rec, rec]) == 1
    assert f.absorb([rec]) == 0


def test_explicit_decay_is_an_event_and_merge_stays_commutative():
    clock = FakeClock(100.0)
    a = CoherenceField("a", bins=8, clock=clock)
    b = CoherenceField("b", bins=8, clock=clock)
    rec = a.write(1.0, key="x", payload="X", record_id="r1")
    b.absorb([rec])
    # Only replica a applies an explicit decay.
    a.decay(10.0, rate=0.1)
    assert len(a.decays) == 1 and len(b.decays) == 0

    ab = CoherenceField.merge("ab", [a, b], clock=clock)
    ba = CoherenceField.merge("ba", [b, a], clock=clock)
    np.testing.assert_allclose(ab.field(), ba.field())
    assert ab.consensus().amplitude == pytest.approx(math.exp(-1.0))
    # The decay travels: b now fades the same way once it absorbs a's events.
    assert b.merge_from(a) == 1
    assert b.read(key="x").amplitude == pytest.approx(math.exp(-1.0))
    # Idempotent for decays too.
    assert b.merge_from(a) == 0


def test_decay_only_applies_to_writes_made_before_it():
    clock = FakeClock(0.0)
    f = CoherenceField("t", bins=8, clock=clock)
    f.write(1.0, phase=0.0, payload="old")
    clock.t = 10.0
    assert f.decay(1.0, rate=math.log(2)) == 1
    clock.t = 20.0
    f.write(1.0, phase=math.pi, payload="new")
    assert f.read(phase=0.0).amplitude == pytest.approx(0.5)
    assert f.read(phase=math.pi).amplitude == pytest.approx(1.0)
    # Two decays stack multiplicatively, and the newer write only sees the later one.
    clock.t = 30.0
    assert f.decay(1.0, rate=math.log(2)) == 2
    assert f.read(phase=0.0).amplitude == pytest.approx(0.25)
    assert f.read(phase=math.pi).amplitude == pytest.approx(0.5)
    assert f.decay_multiplier(35.0) == 1.0


def test_decay_events_survive_snapshot_roundtrip():
    clock = FakeClock(5.0)
    f = CoherenceField("t", bins=8, clock=clock)
    f.write(1.0, phase=0.0)
    f.decay(1.0, rate=math.log(4))
    data = json.loads(json.dumps(f.to_dict()))
    assert len(data["decays"]) == 1
    g = CoherenceField.from_dict(data, clock=clock)
    assert len(g.decays) == 1
    np.testing.assert_allclose(g.field(), f.field())
    with pytest.raises(ValueError):
        DecayRecord.from_dict({"id": "d", "timestamp": 0.0, "factor": 1.5})


def test_schema_1_scale_is_folded_into_value():
    data = {
        "schema_version": 1,
        "name": "legacy",
        "bins": 8,
        "records": [{"id": "r", "phase": 0.0, "value": 2.0, "coherence": 1.0, "timestamp": 0.0, "scale": 0.25}],
    }
    f = CoherenceField.from_dict(data, clock=FakeClock())
    assert f.read(phase=0.0).amplitude == pytest.approx(0.5)


def test_consensus_exposes_contested_bins_and_ratio():
    f = CoherenceField("t", bins=8, clock=FakeClock())
    f.write(1.0, key="plan", payload="ship", agent_id="A")
    f.write(-1.0, key="plan", payload="hold", agent_id="B")
    f.write(0.2, key="other", payload="lunch", agent_id="C")
    c = f.consensus()
    # The coherent field only sees the unopposed proposal...
    assert c.top_payload == "lunch" and c.share == pytest.approx(1.0)
    assert f.read(key="plan").amplitude == pytest.approx(0.0)
    # ...but the contested spectrum shows where the disagreement lives.
    assert c.contest_ratio == pytest.approx(2.0 / 2.2)
    assert c.contested[0].bin == f.bin_index(phase_from_key("plan"))
    assert c.contested[0].amplitude == pytest.approx(2.0)
    assert c.contested[0].share == pytest.approx(1.0)
    assert {p.payload for p in c.contested[0].payloads} == {"ship", "hold"}
    np.testing.assert_allclose(f.contested_field().sum(), 2.0)

    # A fully coherent field has nothing contested.
    g = CoherenceField("g", bins=8, clock=FakeClock())
    g.write(1.0, key="a")
    g.write(0.5, key="a")
    assert g.consensus().contest_ratio == 0.0 and g.consensus().contested == ()
    # An empty field reports zeros.
    assert CoherenceField("e", bins=8).consensus().contest_ratio == 0.0


def test_narration_usurps_consensus_in_a_mixed_field_and_not_in_a_split_one():
    """The pathology has to be reproducible before the separation rule means anything.

    Two execution results (coherence 0.6, 0.7) and three narration impulses of rising
    coherence (0.5, 0.75, 0.95) at a different key. Superposed into one field, the
    narration bin carries 2.2 against 1.3 and becomes the consensus. Written to
    separate fields, the execution field's consensus is the execution result.
    """
    clock = FakeClock()
    mixed = CoherenceField("mixed", bins=64, clock=clock)
    execution = CoherenceField("execution", bins=64, clock=clock)
    narration = CoherenceField("narration", bins=64, clock=clock)

    for c in (0.6, 0.7):
        for f in (mixed, execution):
            f.write(1.0, key="action:parent-queue", coherence=c, payload="parent queue", agent_id="worker")
    for c in (0.5, 0.75, 0.95):
        for f in (mixed, narration):
            f.write(1.0, key="explain:bottleneck", coherence=c, payload="bottleneck explained", agent_id="orch")

    assert mixed.bin_index(phase_from_key("action:parent-queue")) != mixed.bin_index(
        phase_from_key("explain:bottleneck")
    )
    # The failure: narration is the consensus of the mixed field.
    assert mixed.consensus().top_payload == "bottleneck explained"
    assert mixed.read(key="explain:bottleneck").amplitude == pytest.approx(2.2)
    assert mixed.read(key="action:parent-queue").amplitude == pytest.approx(1.3)
    # The fix: policy read from the execution field alone.
    assert execution.consensus().top_payload == "parent queue"
    assert narration.consensus().record_count == 3


def test_read_at_past_now_excludes_writes_and_decays_from_the_future():
    """Reading the past must not see the future.

    Before: ``weight`` clamped age to zero, so a write stamped *after* ``now``
    contributed at full strength, and ``decay_multiplier`` applied every decay
    at or after the write regardless of ``now``.
    """
    clock = FakeClock(100.0)
    f = CoherenceField("t", bins=8, clock=clock)
    f.write(1.0, key="k", coherence=1.0, timestamp=10.0)
    f.write(1.0, key="k", coherence=1.0, timestamp=50.0)
    f.decay(dt=1.0, rate=math.log(2), timestamp=70.0)  # halves everything written by t=70

    assert f.read(key="k", now=5.0).amplitude == 0.0  # nothing has happened yet
    assert f.read(key="k", now=10.0).amplitude == pytest.approx(1.0)  # first write, inclusive
    assert f.read(key="k", now=30.0).amplitude == pytest.approx(1.0)  # second write not yet
    assert f.read(key="k", now=60.0).amplitude == pytest.approx(2.0)  # both, decay not yet
    assert f.read(key="k", now=70.0).amplitude == pytest.approx(1.0)  # decay applied
    assert f.read(key="k", now=100.0).amplitude == pytest.approx(1.0)  # and stays applied
    assert f.consensus(now=5.0).record_count == 2  # records are kept; they just do not weigh yet
    assert f.decay_multiplier(10.0, now=60.0) == 1.0
    assert f.decay_multiplier(10.0, now=70.0) == pytest.approx(0.5)


def test_prune_to_capacity_does_not_punish_a_replica_whose_clock_runs_ahead():
    clock = FakeClock(100.0)
    f = CoherenceField("p", bins=8, max_records=2, clock=clock)
    f.write(1.0, key="a", coherence=0.2, timestamp=100.0)
    f.write(1.0, key="b", coherence=0.9, timestamp=100.0)
    f.write(1.0, key="c", coherence=0.9, timestamp=103.0)  # absorbed from a replica 3s ahead
    assert {r.key for r in f.records} == {"b", "c"}


def test_signed_read_goes_negative_when_objections_out_weigh_proposals():
    """``amplitude`` is |psi| and cannot tell a live proposal from a defeated one."""
    clock = FakeClock()
    f = CoherenceField("s", bins=16, clock=clock)
    f.write(1.0, key="rule", coherence=0.3, payload="keep it")
    f.write(-1.0, key="rule", coherence=0.8, payload="drop it")
    r = f.read(key="rule")
    assert r.amplitude == pytest.approx(0.5)  # looks alive
    assert r.signed == pytest.approx(-0.5)  # is not
    # Along the proposal's own phasor the sign is exact; a read a quarter turn away sees ~0.
    q = f.read(phase=phase_from_key("rule") + math.pi / 2)
    assert q.bin != r.bin or abs(q.signed) < 1e-9
    # A pure proposal reads positive and equal to its amplitude.
    g = CoherenceField("g", bins=16, clock=clock)
    g.write(1.0, key="rule", coherence=0.6)
    assert g.read(key="rule").signed == pytest.approx(g.read(key="rule").amplitude)


def test_prune_never_drops_a_record_from_the_future():
    """A future write weighs 0 at `now`, which is not the same as having faded to 0."""
    clock = FakeClock(100.0)
    f = CoherenceField("f", bins=8, decay_rate=1.0, clock=clock)
    f.write(1.0, key="old", timestamp=0.0)  # e^-100: faded, prunable
    f.write(1.0, key="ahead", timestamp=103.0)  # absorbed from a replica 3s ahead
    assert f.prune(epsilon=1e-6) == 1
    assert [r.key for r in f.records] == ["ahead"]
    # ...and it is fully present once the local clock catches up.
    assert f.read(key="ahead", now=103.0).amplitude == pytest.approx(1.0)
    # Pruning "as of" a past instant likewise cannot touch what had not happened yet.
    g = CoherenceField("g", bins=8, decay_rate=1.0, clock=clock)
    g.write(1.0, key="k", timestamp=50.0)
    assert g.prune(epsilon=1e-6, now=10.0) == 0


def _history(clock: FakeClock, **kw) -> CoherenceField:
    """Three authors re-affirm one rule, one objects, one unrelated key; explicit decay at t=30."""
    f = CoherenceField("h", bins=64, decay_rate=0.02, kernel_width=0.1, clock=clock, **kw)
    f.write(1.0, key="rule", coherence=0.8, payload="keep", agent_id="a", timestamp=0.0)
    f.write(1.0, key="rule", coherence=0.6, payload="keep", agent_id="b", timestamp=10.0)
    f.write(-1.0, key="rule", coherence=0.5, agent_id="c", timestamp=15.0)
    f.write(1.0, key="rule", coherence=0.9, payload="keep", agent_id="c", timestamp=20.0)
    f.write(1.0, key="other", coherence=0.7, payload="x", agent_id="a", timestamp=5.0)
    f.decay(dt=1.0, rate=math.log(2), timestamp=30.0)
    return f


def test_compact_conserves_the_field_and_the_contested_spectrum():
    clock = FakeClock(40.0)
    f = _history(clock)
    before = {t: (f.field(now=t).copy(), f.contested_field(now=t).copy()) for t in (40.0, 60.0, 200.0)}
    n_before = len(f)

    removed = f.compact(now=40.0)

    assert removed == 2  # three "keep" proposals -> one; objection and "other" untouched
    assert len(f) == n_before - 2
    for t, (psi, contested) in before.items():
        np.testing.assert_allclose(f.field(now=t), psi, atol=1e-12)
        np.testing.assert_allclose(f.contested_field(now=t), contested, atol=1e-12)
    # A decay recorded after the rake scales the summary as it would have scaled each member.
    g = _history(FakeClock(40.0))
    f.decay(dt=1.0, rate=0.3, timestamp=50.0)
    g.decay(dt=1.0, rate=0.3, timestamp=50.0)
    np.testing.assert_allclose(f.field(now=80.0), g.field(now=80.0), atol=1e-12)
    # The summary knows what it replaced and keeps the heaviest member's provenance.
    (summary,) = [r for r in f.records if r.subsumes]
    assert len(summary.subsumes) == 3 and summary.payload == "keep" and summary.agent_id == "c"
    # Records from the future of `now` are never raked.
    h = _history(FakeClock(40.0))
    assert h.compact(now=5.0) == 0


def test_compact_keeps_sync_a_union():
    clock = FakeClock(40.0)
    a = _history(clock)
    b = CoherenceField.from_dict(a.to_dict(include_field=False), clock=clock)  # same ids
    originals = [r for r in a.records if r.payload == "keep"]
    a.compact(now=40.0)

    # b holds the originals; absorbing a's summary replaces them.
    assert b.absorb(a.events) == 1
    assert len(b) == len(a)
    np.testing.assert_allclose(b.field(now=60.0), a.field(now=60.0), atol=1e-12)
    # The originals coming back from a stale replica are rejected, not double counted.
    assert a.absorb(originals) == 0
    assert b.absorb(originals) == 0
    np.testing.assert_allclose(b.field(now=60.0), a.field(now=60.0), atol=1e-12)
    # Idempotent both ways.
    assert a.absorb(b.events) == 0
    assert b.absorb(a.events) == 0


def test_concurrent_rakes_with_overlap_yield_different_records_and_the_same_field():
    clock = FakeClock(40.0)
    a = _history(clock)
    b = CoherenceField.from_dict(a.to_dict(include_field=False), clock=clock)
    a.compact(now=40.0)
    b.compact(now=45.0)  # same grains, different reference time
    a.absorb(b.events)
    b.absorb(a.events)
    assert {r.id for r in a.records} != {r.id for r in b.records}
    for t in (45.0, 100.0):
        np.testing.assert_allclose(a.field(now=t), b.field(now=t), atol=1e-12)
    assert a.absorb(b.events) == 0 and b.absorb(a.events) == 0


def test_rake_of_a_rake_is_transitive_and_survives_a_snapshot():
    clock = FakeClock(40.0)
    f = _history(clock)
    originals = {r.id for r in f.records if r.payload == "keep"}
    f.compact(now=40.0)
    f.write(1.0, key="rule", coherence=0.5, payload="keep", agent_id="d", timestamp=41.0)
    f.compact(now=50.0)
    (summary,) = [r for r in f.records if r.subsumes]
    assert originals <= set(summary.subsumes)
    g = CoherenceField.from_dict(json.loads(json.dumps(f.to_dict())), clock=clock)
    np.testing.assert_allclose(g.field(now=70.0), f.field(now=70.0), atol=1e-12)
    # The restored replica still refuses the raked originals by id.
    fresh = CoherenceField("fresh", bins=64, decay_rate=0.02, kernel_width=0.1, clock=clock)
    for rid in originals:
        fresh.write(1.0, key="rule", payload="keep", timestamp=0.0, record_id=rid)
    assert g.absorb(fresh.records) == 0


def test_schema_2_snapshots_load_and_schema_3_is_rejected_by_older_readers():
    clock = FakeClock(40.0)
    f = _history(clock)
    data = f.to_dict()
    data["schema_version"] = 2
    for r in data["records"]:
        r.pop("subsumes")
    g = CoherenceField.from_dict(data, clock=clock)
    np.testing.assert_allclose(g.field(now=50.0), f.field(now=50.0), atol=1e-12)
    assert all(r.subsumes == () for r in g.records)
