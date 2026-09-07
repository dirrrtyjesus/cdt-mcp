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
