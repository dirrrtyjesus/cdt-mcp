"""harmonic_field_read: reading a manuscript as a coherence field.

This rewrites the two parts of ``harmonic_read.py`` whose docstrings
outran their execution, and then does the thing the file was reaching
for: it reads the manuscript *through* a :class:`CoherenceField`.

What changed and why
--------------------
* ``read_adaptive`` walked the scales smallest-first and returned the
  first view that fit, so it always returned ``quantum``. It now walks
  largest-first and returns the highest-resolution view within budget.
* "Resonance" was the Hamming distance between two SHA-256 hashes. SHA-256
  is built so that nothing about the input survives into the output; the
  distance between hashes of any two strings is Binomial(64, 1/2) noise
  (mean 0.50, sd 0.06) and 97% of all entries cleared the 0.618 threshold
  for every query. ``core.phase_from_key`` uses the same primitive as an
  *address* -- same key, same bin -- and never asks whether two hashes are
  close. A hash can say where something is, never what it is like.
  Resonance is now **MinHash** over word shingles: still a hash, still
  content-addressable, but agreement between two signatures is an unbiased
  estimate of the Jaccard similarity of the underlying shingle sets. The
  word "resonance" is now true.
* The honest phase in the original was ``temporal_stream``'s
  ``i / N * 2*pi`` -- position in the manuscript on the ring. That is the
  phase each entry is written at here. ``coherence`` is its resonance with
  the query, ``payload`` its index. ``consensus()`` then returns the
  *region* of the manuscript that resonates, and the contested spectrum
  shows where the query lands on entries that argue with each other.

Homomorphic superposition
-------------------------
"Writes never overwrite; they superpose" is a statement about addition,
and addition is what homomorphic encryption does natively. ``fhe_superpose``
uses CKKS (via TenSEAL) so that each writer encrypts its own impulse
under the reader's public key, an aggregator that holds no key sums the
ciphertexts, and the reader decrypts a field equal to the plaintext one.
Encryption preserves the *function*: it does not make a meaningless
resonance meaningful, and it does not make ``consensus()`` (an argmax --
nonlinear) cheap. So the split is: writers score resonance in the clear
against a public query, superposition happens under encryption where no
party sees any other party's impulse, and reading requires the key.
"""

from __future__ import annotations

import hashlib
import re
import struct
import sys
from dataclasses import dataclass

import numpy as np

from cdt_mcp.core import TWO_PI, CoherenceField, ConsensusResult

# ---------------------------------------------------------------- scales

TEMPORAL_SCALES = {
    "quantum": 500,
    "cellular": 2_000,
    "network": 8_000,
    "ecosystem": 32_000,
    "geological": 128_000,
}


def pick_scale(views: dict[str, str], budget: int) -> str:
    """Highest-resolution view whose rendered size fits ``budget``.

    Walks largest-first; falls back to the smallest view if nothing fits.
    (The original walked smallest-first and returned the first fit, which
    is always the smallest.)
    """
    for scale in reversed(TEMPORAL_SCALES):
        if len(views[scale]) <= budget:
            return scale
    return next(iter(TEMPORAL_SCALES))


# --------------------------------------------------------------- minhash

_WORD = re.compile(r"[a-z0-9]+")
_MASK = (1 << 61) - 1  # Mersenne prime; the hash family is (a*x + b) mod p


_STOP = {"a", "an", "the", "is", "are", "of", "to", "and", "or", "in", "on", "at", "how", "does", "do", "so", "no"}


def shingles(text: str, ks: tuple[int, ...] = (1, 2)) -> set[int]:
    """Word 1- and 2-shingles (stopwords dropped), hashed to 64-bit ints.

    Short queries against long entries need unigrams to overlap at all;
    bigrams keep phrase identity in the estimate.
    """
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    out: set[int] = set()
    for k in ks:
        for i in range(max(0, len(words) - k + 1)):
            h = hashlib.blake2b(" ".join(words[i : i + k]).encode(), digest_size=8).digest()
            out.add(struct.unpack("<Q", h)[0])
    return out


class MinHasher:
    """``n`` universal hash functions; a signature is the min under each."""

    def __init__(self, n: int = 128, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _MASK, size=n, dtype=np.int64)
        self.b = rng.integers(0, _MASK, size=n, dtype=np.int64)

    def signature(self, text: str) -> np.ndarray:
        xs = np.fromiter(shingles(text), dtype=np.uint64)
        if xs.size == 0:
            return np.full(self.a.size, _MASK, dtype=np.int64)
        # (a*x + b) mod p, done in Python ints to avoid overflow on 61-bit products.
        sig = np.empty(self.a.size, dtype=np.int64)
        for j in range(self.a.size):
            a, b = int(self.a[j]), int(self.b[j])
            sig[j] = min(((a * int(x) + b) % _MASK) for x in xs)
        return sig

    @staticmethod
    def resonance(s1: np.ndarray, s2: np.ndarray) -> float:
        """Fraction of agreeing slots: an unbiased estimate of Jaccard(A, B)."""
        return float(np.mean(s1 == s2))


# ------------------------------------------------------------ field read


@dataclass(frozen=True)
class FieldRead:
    consensus: ConsensusResult
    scores: tuple[tuple[int, float], ...]  # (entry index, resonance), sorted desc
    field: CoherenceField


def entry_text(entry: dict) -> str:
    return f"{entry.get('prompt', '')}\n{entry.get('composition', '')}"


def field_read(
    entries: list[dict],
    query: str,
    *,
    hasher: MinHasher | None = None,
    bins: int = 64,
    kernel_width: float = 0.0,
    floor: float = 0.0,
) -> FieldRead:
    """Write every entry into a field at phase ``i/N * 2pi`` with coherence
    equal to its MinHash resonance with ``query``; read consensus.

    ``floor`` drops impulses below a resonance threshold; leave it at 0 to
    let the field decide (weak impulses just contribute little energy).
    """
    h = hasher or MinHasher()
    q = h.signature(query)
    n = len(entries)
    field = CoherenceField("manuscript", bins=bins, kernel_width=kernel_width, clock=lambda: 0.0)
    scores = []
    for i, e in enumerate(entries):
        r = MinHasher.resonance(q, h.signature(entry_text(e)))
        scores.append((i, r))
        if r > floor:
            field.write(1.0, phase=i / n * TWO_PI, coherence=r, agent_id=f"entry-{i}", payload=str(i), timestamp=0.0)
    scores.sort(key=lambda t: -t[1])
    return FieldRead(consensus=field.consensus(top_k=3), scores=tuple(scores), field=field)


# ------------------------------------------------------------------ FHE


def impulse_vector(field: CoherenceField, rec_id: str) -> np.ndarray:
    """One writer's contribution to the field, as a real vector [re..., im...]."""
    rec = field._records[rec_id]  # noqa: SLF001  (demo: reach in for one record's term)
    w = rec.coherence * rec.value
    k = field._kernel(rec.phase)  # noqa: SLF001
    z = w * np.exp(1j * rec.phase) * k
    return np.concatenate([z.real, z.imag])


def fhe_superpose(field: CoherenceField) -> tuple[np.ndarray, np.ndarray, dict]:
    """Superpose every writer's impulse under CKKS; return (decrypted, plaintext, info).

    Roles: each *writer* encrypts only its own impulse with the reader's
    public key; the *aggregator* holds ciphertexts and no secret key and
    just adds; the *reader* decrypts once.
    """
    import tenseal as ts

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    public = ctx.copy()
    public.make_context_public()  # what writers and the aggregator get: no secret key

    # writers
    cipher_impulses = [ts.ckks_vector(public, impulse_vector(field, rid).tolist()) for rid in field._records]  # noqa: SLF001
    # aggregator: addition only, on a context with no secret key
    agg = cipher_impulses[0]
    for c in cipher_impulses[1:]:
        agg = agg + c
    assert not public.has_secret_key()
    # reader
    decrypted = np.asarray(agg.decrypt(ctx.secret_key()))
    plain = field.field(now=0.0)
    plain_vec = np.concatenate([plain.real, plain.imag])
    info = {
        "writers": len(cipher_impulses),
        "ciphertext_bytes_each": len(cipher_impulses[0].serialize()),
        "max_abs_error": float(np.max(np.abs(decrypted - plain_vec))),
    }
    return decrypted, plain_vec, info


# ------------------------------------------------------------------ demo


def _manuscript() -> list[dict]:
    topics = {
        "decay": "explicit decay is an event; replicas that decayed at different moments still converge",
        "phase": "phase bins are linspace with endpoint false so bin zero and the last bin no longer alias",
        "contest": "opposing proposals cancel in the coherent field and show up in the contested spectrum",
        "merge": "merge is a set union of write events: idempotent commutative associative",
        "noise": "the quick brown fox jumps over the lazy dog near the river bank at dawn",
    }
    order = [
        "noise",
        "noise",
        "decay",
        "decay",
        "phase",
        "contest",
        "contest",
        "merge",
        "noise",
        "decay",
        "decay",
        "noise",
    ]
    return [{"prompt": f"τ{i} on {t}", "composition": topics[t] + f" (entry {i})"} for i, t in enumerate(order)]


def main() -> None:
    entries = _manuscript()
    query = "how does explicit decay keep replicas converging"
    h = MinHasher()

    # 1. resonance now tracks content
    q = h.signature(query)
    near = h.signature(query + ".")
    far = h.signature("purple monkeys dishwasher tuesday")
    print(f"resonance(query, query+'.') = {MinHasher.resonance(q, near):.2f}")
    print(f"resonance(query, unrelated) = {MinHasher.resonance(q, far):.2f}")

    # 2. the field read
    fr = field_read(entries, query, kernel_width=0.15)
    c = fr.consensus
    print(f"\nquery: {query!r}")
    print("top resonances:", [(i, round(r, 2)) for i, r in fr.scores[:4]])
    print(
        f"consensus bin {c.bin} at phase {c.phase:.2f} -> entries",
        [pw.payload for pw in c.payloads if pw.weight > 0.01],
    )
    print(f"share {c.share:.2f}, confidence {c.confidence:.1f}x, contest ratio {c.contest_ratio:.2f}")
    region = [int(pw.payload) for alt in c.alternatives for pw in alt.payloads]
    print("resonant region of the manuscript:", sorted(set(region)))

    # 3. superposition under encryption (optional: pip install tenseal)
    try:
        dec, plain, info = fhe_superpose(fr.field)
    except ImportError:
        print("\nCKKS demo skipped: `pip install tenseal` to run the encrypted superposition")
        return
    print(
        f"\nCKKS: {info['writers']} writers, {info['ciphertext_bytes_each'] // 1024} KiB per ciphertext, "
        f"max |decrypted - plaintext| = {info['max_abs_error']:.2e}"
    )
    half = len(dec) // 2
    psi_enc = dec[:half] + 1j * dec[half:]
    print("argmax |ψ| from the decrypted field:", int(np.argmax(np.abs(psi_enc))), "== consensus bin", c.bin)


if __name__ == "__main__":
    sys.exit(main())
