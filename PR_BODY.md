## The finding first

Put an orchestrator's explanations and its workers' execution results into **one** field and, around t=27, the explanations become the consensus. Three explanations of rising coherence (0.50 → 0.75 → 0.95) out-accumulate two execution results, and the field faithfully reports that the most coherent energy in the campaign is the orchestrator talking about the bottleneck. That is the sentence from the postmortem — "my explanations improved while my execution stayed substantially the same" — reproduced inside a CDT rather than narrated by one.

The field does not fix that on its own; it exposes it. The fix is architectural and foundational: narration gets its own field, policy is read only from the execution field, and the two can no longer impersonate each other.

## What is in `astra_campaign/`

The `astra_campaign/` directory contains two complementary studies of coherence fields:

### 1. `astra_campaign.py` — routing a campaign vs. reading it

Replays a ten-result refactor campaign in front of three observers:

- **`Router`** — last-write-wins. Each incoming result is a local task; adopt it, route on. It flips policy six times in ten results and sees zero disagreement, because a worker's negative result looks like any other result to it.
- **one `CoherenceField`** — everything superposed, narration included. Correct final answer, but narration briefly becomes the consensus at t=27.
- **execution field + narration field** — consensus lands on `parent-owned-queue` with 0.91 of coherent energy at the peak (58× the mean bin) and reports what the router structurally cannot: bin 14, `workers-launch-broad-suites`, net −0.25 from 4 results by `worker-2` and `worker-3`, with 0.47 of all impulse energy cancelled. The disagreement is held and legible instead of routed away.

`--plot campaign_ring.png` renders the execution field as a polar bar chart: coherent |ψ| (what `consensus()` reads) against the contested spectrum Σ|w| − |Σw| (what would have vanished from a merge).

```bash
python astra_campaign/astra_campaign.py --plot astra_campaign/campaign_ring.png
```

### 2. `harmonic_field_read.py` — reading a manuscript as a field

Reworks `harmonic_read.py` to ground semantic coherence in `cdt_mcp.core`:
- **MinHash resonance**: Replaces SHA-256 Hamming distance noise with MinHash over word 1- and 2-shingles, giving an unbiased estimate of Jaccard similarity.
- **Adaptive scale resolution**: Walks largest-first to return the highest-resolution view fitting the budget rather than collapsing prematurely to `quantum`.
- **Homomorphic superposition**: Uses CKKS (via TenSEAL) to prove that wave superposition maps natively to homomorphic addition. Writers encrypt impulse vectors under the reader's public key; an untrusted aggregator sums ciphertexts; the reader decrypts the identical field (max error ~8.65e-9) and argmax bin.

```bash
python astra_campaign/harmonic_field_read.py
```

## Foundational design added to `THEORY.md`

The separation principle is now formalized in `docs/THEORY.md` under **"Foundational design: separation of narration and execution"**:
- **Field orthogonality**: Narration and execution are partitioned into disjoint coherence fields (`campaign:execution` and `campaign:narration`).
- **Policy strictly from execution**: Operational decisions and routing depend exclusively on the execution field.
- **Narration as diagnostic state**: Tracks explanation quality and agent understanding with zero voting weight on operational state.
- **Disagreement legibility**: Conflicting results stay sharp in the execution field's contested spectrum ($C_k = A_k - |\Psi_k|$).

## Accompanying updates

- **`README.md`**: Added a pointer to `astra_campaign/` alongside `examples/multi_agent_merge.py`, and added the separation of narration and execution guarantee to `Semantics and guarantees`.
- **`tests/test_core.py`**: Added `test_narration_and_execution_field_separation` verifying that high-coherence narrative explanations in an orthogonal field do not pollute execution consensus.
- Passes `pytest`, `ruff check .`, and `mypy src`.
