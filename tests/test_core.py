import json
import math

import numpy as np
import pytest

from cdt_mcp.core import CoherenceField, WriteRecord, normalize_phase, phase_from_key


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
    assert data["schema_version"] == 1
    assert len(data["field"]) == 32
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
