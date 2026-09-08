# astra_campaign

Two readings of a coherence field, both against `cdt_mcp.core`.

## `astra_campaign.py` — routing a campaign vs. reading it

A last-write-wins orchestrator (`Router`) and two `CoherenceField` readers watch
the same ten-result refactor campaign. The router flips policy on every arrival
and cannot see a worker's negative result as disagreement. The execution field
lands on `parent-owned-queue` with 0.91 of coherent energy at the peak and
reports the `workers-launch-broad-suites` bin as *contested*: net −0.25 from
four results by two workers, 0.47 of all impulse energy cancelled. The
disagreement is held, not routed away.

The finding: with narration and execution in **one** field, the orchestrator's
three improving explanations briefly *become the consensus* at t=27. That is the
postmortem sentence — "my explanations improved while my execution stayed the
same" — reproduced inside a CDT. Narration gets its own field; policy is read
only from execution.

```
python astra_campaign.py --plot campaign_ring.png
```

Output: `astra_campaign_output.txt`, `campaign_ring.png` (blue = coherent |ψ|,
what `consensus()` reads; orange = the contested spectrum Σ|w| − |Σw|, what a
merge would have erased).

## `harmonic_field_read.py` — reading a manuscript as a field

Reworks `harmonic_read.py` where its docstrings outran its execution, then reads
the manuscript through a field:

* **Resonance** was Hamming distance between SHA-256 hashes — Binomial(64, ½)
  noise; 97% of entries "resonated" with every query. It is now MinHash over
  word shingles, so agreement estimates Jaccard similarity. Same primitive as
  `phase_from_key`, opposite semantics: a hash can say *where* something is,
  never *what it is like*.
* **`read_adaptive`** walked scales smallest-first and always returned
  `quantum`. It now returns the highest-resolution view within budget.
* **The field read** writes each entry at phase `i/N·2π` with `coherence` equal
  to its resonance with the query; `consensus()` returns the resonant *region*
  of the manuscript, and the contested spectrum shows where the query lands on
  entries that argue with each other.
* **Homomorphic superposition.** "Writes never overwrite; they superpose" is a
  statement about addition, and addition is what homomorphic encryption does
  natively. With `pip install tenseal`, each writer encrypts its own impulse
  under the reader's CKKS public key, an aggregator holding no secret key sums
  the ciphertexts, and the reader decrypts a field equal to the plaintext one
  (max error ~1e-8; argmax matches `consensus().bin`). Encryption preserves the
  function — it cannot make a meaningless resonance meaningful — and it makes
  addition cheap and argmax expensive, so superposition happens sealed and
  reading requires the key.

```
python harmonic_field_read.py
```

Output: `harmonic_field_read_output.txt`.

Both scripts import `cdt_mcp.core`; run from the repo root with
`pip install -e .` or `PYTHONPATH=src`.

## `field_claude_md.py` — a CLAUDE.md rendered from a field

The "catastrophic remembering" ratchet (Chakrabarti 2026): instructions are cheap
to add and expensive to delete, and the rationale that justified each one decays
faster than the instruction. A field dissolves both halves.

* An instruction is an impulse with a half-life (40 commits here). It stays alive
  only by **re-affirmation** — the failure recurs, someone writes it again, the
  impulse re-energizes. Nothing is deleted, so there is no O(2^|D|) regression
  risk; nothing is immortal, so there is no fossil layer.
* The rationale (failure / hypothesis / outcome) is written to a **second field at
  the same key**. The model renders the instruction field; the maintainer renders
  the rationale field. Same address, two readers — the paper's hidden-comment fix
  with the coupling made structural.
* `contest` writes `value=-1` at the key. Against a fresh rule it flags the line
  `<!-- contested -->`; against a stale one it wins, and the objection stays in
  the rationale field as the record of why.

Over a 160-commit history the append-only file holds 5 instructions forever; the
field-rendered one ends at 3 live from 11 records kept, 0 deleted.

```
python field_claude_md.py --plot ratchet.png
```

Output: `field_claude_md_output.txt`, `ratchet.png`.

Two things this surfaced in `core.py`, now fixed there with tests that fail on the
previous tree: `read(now=past)` counted records with `timestamp > now` at full
strength (age was clamped to 0), so the demo had to replay history instead of
reading the past — it now writes the history once and rewinds; and per-key
`amplitude` is unsigned, so an objection that out-weighs its rule read as energy —
`ReadResult.signed` is the field's component along the read phasor, and the demo
uses it. A third followed from the fix: `prune()` treated a future write's weight of
0 as "faded" and deleted it — it now only considers records that have happened.

Keys here get explicit slots on the ring rather than `phase_from_key` hashes.
Hashing is right for open-world swarms (any replica maps a key to the same bin
with no coordination) but collides at ~n²/2B: 50 instructions in 4096 bins share
a bin about 30% of the time. A CLAUDE.md is a closed, small keyspace, so slots
are assigned in registration order and are collision-free up to `bins`; the
registry is the one thing replicas must share.
