# Coherence Data Types: model and guarantees

## The primitive

A CDT is a ring of `N` phase bins `θ_k = 2πk/N`. Each write is an impulse

```
ψ_i = c_i · v_i · e^{iθ_i}          c_i ∈ [0,1] (coherence),  v_i ∈ ℝ (value),  θ_i ∈ [0,2π)
```

deposited at the nearest bin (or spread over neighbours with a wrapped Gaussian of width `σ`
when `kernel_width > 0`). The field is the superposition

```
Ψ(t) = Σ_i  ψ_i · e^{-λ (t - t_i)} · s_i
```

where `λ` is the field's continuous `decay_rate` and `s_i` is any explicit scaling applied by
`decay()`. Reading at a phase returns `|Ψ_k|`. Consensus is

```
k* = argmax_k |Ψ_k|
confidence = |Ψ_k*| / mean_k |Ψ_k|        share = |Ψ_k*| / Σ_k |Ψ_k|
```

`share` is the quantity to act on: it is 1.0 when a single bin holds all the energy and tends to
`1/N` when proposals are spread evenly.

## Why write events are the source of truth

Raw field addition `Ψ_a + Ψ_b` is commutative and associative but **not idempotent**: syncing the
same snapshot twice doubles its energy. That breaks replication under retries and gossip.

cdt-mcp therefore stores the set of write events `E = {e_i}` (each with a UUID) and derives `Ψ`
from `E`. Merge is `E_a ∪ E_b`, a grow-only set, which is a join-semilattice: idempotent,
commutative, associative. Two replicas that have seen the same events render identical fields,
independent of order or duplication. In CRDT terms the *transport* is a G-Set; the *readout* is
the wave-superposition consensus that makes CDTs different from CRDTs.

Decay is folded into the derivation (`e^{-λ(t - t_i)}`), so it needs no clock synchronization
between replicas beyond agreeing on wall-clock time to within the decay time-scale.

## Where CDTs differ from CRDTs

| | CRDT (e.g. LWW-register, OR-set) | CDT |
|---|---|---|
| Conflict | detected, resolved by order/ID | never detected; proposals coexist |
| Result | one winner, deterministic but arbitrary | a field; winner = highest density, losers remain readable |
| Disagreement | not expressible | negative `value` interferes destructively |
| Confidence | none | `coherence` scales each impulse; `share` reports contestedness |
| Forgetting | tombstones / GC | continuous decay |

## Interference

Two impulses in the same bin add as complex numbers with (almost) the same phase, so agreement is
constructive. A negative `value` flips the sign and cancels. With `kernel_width > 0`, impulses at
nearby phases partially overlap, so reads between two proposals return an interpolated blend.

## Limits

* **Key collisions.** Keys hash to phases; with `N` bins two random keys share a bin with
  probability ≈ `1/N`. Use more bins or explicit phases for many-keyed fields.
* **`max_records` pruning** drops the weakest events when a field exceeds its cap. This is the
  one non-monotonic operation and can make replicas diverge. Prefer decay + `prune()` and a
  generous cap.
* **`tau_k`** is metadata averaged by spectral density on merge. It is informative, not a
  lattice element; don't rely on it for convergence.
* **Field rendering is O(records × bins)**. With the default 64 bins and 10,000 records a
  consensus read is well under a millisecond in NumPy; raise `bins` with that in mind.

## Lineage

The CDT idea comes from Fractal Harmonic Processing: state as a coherence field, truth as the
region of highest spectral density, memory as accumulated phase-locks that decay when unattended.
See the [fhp-computing](https://github.com/dirrrtyjesus/fhp-computing) repository, in particular
*Ublox: The Gamification of Resonance* and the PTO × CDT synthesis, whose original
`CoherenceDataType` prototype this package composes with.
