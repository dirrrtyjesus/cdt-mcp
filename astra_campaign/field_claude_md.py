"""A CLAUDE.md that is rendered from a coherence field instead of edited.

The ratchet in agentic instruction files ("catastrophic remembering",
Chakrabarti 2026) has two halves: instructions are cheap to add and
expensive to delete, and the rationale that justified each one decays
faster than the instruction itself. A field dissolves both halves at once.

* An instruction is an impulse, not a line. It fades at
  ``exp(-decay_rate * age)`` unless the failure recurs and someone writes
  it again, which re-energizes it. Nothing is ever deleted, so there is no
  O(2^|D|) regression risk from deleting the wrong thing; nothing is
  immortal either, so there is no fossil layer. Deletion becomes the
  absence of re-affirmation.
* The rationale is written to a *second* field at the *same key*. The
  executing model renders the instruction field; the maintainer renders
  the rationale field. Same address, two readers. The paper's
  hidden-comment fix, with the coupling made structural.
* Disagreement is not deletion. ``contest`` writes ``value=-1`` at the
  instruction's key; the rendered file flags it as contested instead of
  silently dropping it, and the rationale field carries the objection.

Run:  python field_claude_md.py [--plot ratchet.png]
"""

from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass

from cdt_mcp.core import CoherenceField, phase_from_key

HALF_LIFE = 40.0  # commits: an un-affirmed instruction loses half its energy every 40 commits
DECAY = math.log(2) / HALF_LIFE
BINS = 1024  # enough that a handful of keys do not collide; collisions are checked anyway


@dataclass(frozen=True)
class Line:
    key: str
    text: str
    energy: float
    contested: float
    age: float
    affirmations: int
    authors: tuple[str, ...]


class FieldClaudeMd:
    """Two fields, one keyspace. ``instructions`` is what the model reads;
    ``rationale`` is what the maintainer reads."""

    def __init__(self, *, half_life: float = HALF_LIFE, bins: int = BINS) -> None:
        self._now = 0.0
        clock = lambda: self._now  # noqa: E731
        rate = math.log(2) / half_life
        self.instructions = CoherenceField("claude.md:instructions", bins=bins, decay_rate=rate, clock=clock)
        self.rationale = CoherenceField("claude.md:rationale", bins=bins, decay_rate=rate, clock=clock)
        self._bins: dict[int, str] = {}

    # ------------------------------------------------------------ writes

    def _claim(self, key: str) -> None:
        b = self.instructions.bin_index(phase_from_key(key))
        owner = self._bins.setdefault(b, key)
        if owner != key:
            raise ValueError(f"key {key!r} collides with {owner!r} in bin {b}; raise bins")

    def affirm(
        self,
        key: str,
        instruction: str,
        *,
        failure: str,
        hypothesis: str,
        outcome: str,
        author: str,
        commit: float,
        coherence: float = 1.0,
    ) -> None:
        """Add or re-affirm an instruction, with the rationale that justifies it.

        Re-affirming an existing key is the *only* way an instruction stays
        alive: it superposes a fresh impulse on the decayed one.
        """
        self._claim(key)
        self._now = commit
        self.instructions.write(
            1.0, key=key, coherence=coherence, agent_id=author, payload=instruction, timestamp=commit
        )
        self.rationale.write(
            1.0,
            key=key,
            coherence=coherence,
            agent_id=author,
            payload=f"failure: {failure} | hypothesis: {hypothesis} | outcome: {outcome}",
            timestamp=commit,
        )

    def contest(self, key: str, *, reason: str, author: str, commit: float, coherence: float = 1.0) -> None:
        """Argue against an instruction without deleting it."""
        self._claim(key)
        self._now = commit
        self.instructions.write(-1.0, key=key, coherence=coherence, agent_id=author, timestamp=commit)
        self.rationale.write(
            1.0, key=key, coherence=coherence, agent_id=author, payload=f"objection: {reason}", timestamp=commit
        )

    # ------------------------------------------------------------- reads

    def lines(self, *, now: float, floor: float = 0.25) -> list[Line]:
        """Instructions with net energy above ``floor`` (in units of one fresh, fully coherent write)."""
        self._now = now
        contested = self.instructions.contested_field(now=now)
        out: list[Line] = []
        for b, key in self._bins.items():
            r = self.instructions.read(key=key, now=now)
            # Every write at this key shares one phase, so psi[bin] = net * e^{i phase}.
            # Project back onto the key's phasor for a *signed* net: amplitude alone
            # would count an objection that out-weighs its rule as "live".
            net = (r.real + 1j * r.imag) * cmath.exp(-1j * phase_from_key(key))
            pos = [rec for rec in self.instructions.records if rec.key == key and rec.value > 0]
            latest = max(pos, key=lambda rec: rec.timestamp)
            if net.real < floor:
                continue
            out.append(
                Line(
                    key=key,
                    text=latest.payload or "",
                    energy=float(net.real),
                    contested=float(contested[b]),
                    age=now - latest.timestamp,
                    affirmations=len(pos),
                    authors=tuple(sorted({rec.agent_id for rec in pos if rec.agent_id})),
                )
            )
        out.sort(key=lambda ln: -ln.energy)
        return out

    def render(self, *, now: float, floor: float = 0.25) -> str:
        """What the executing model sees: instructions only, no rationale."""
        body = []
        for ln in self.lines(now=now, floor=floor):
            flag = "  <!-- contested -->" if ln.contested > 0.05 else ""
            body.append(f"- {ln.text}{flag}")
        return f"# CLAUDE.md  (rendered at commit {now:.0f}, {len(body)} live instructions)\n" + "\n".join(body)

    def render_rationale(self, *, now: float, floor: float = 0.25) -> str:
        """What the maintainer sees: the same keys, with why."""
        out = [f"# rationale  (commit {now:.0f})"]
        for ln in self.lines(now=now, floor=floor):
            recs = sorted((r for r in self.rationale.records if r.key == ln.key), key=lambda r: r.timestamp)
            out.append(
                f"\n## {ln.key}  energy {ln.energy:.2f}  contested {ln.contested:.2f}  "
                f"age {ln.age:.0f}  affirmed x{ln.affirmations} by {', '.join(ln.authors)}"
            )
            for r in recs:
                out.append(f"   @{r.timestamp:.0f} {r.agent_id}: {r.payload}")
        return "\n".join(out)


# ------------------------------------------------------------------ demo


Event = tuple  # ("affirm" | "contest", commit, kwargs)

HISTORY: list[Event] = [
    (
        "affirm",
        0,
        dict(
            key="tests:run-before-commit",
            instruction="Run `pytest -q` before every commit.",
            failure="pushed a red suite twice in week one",
            hypothesis="agent skips tests when not told",
            outcome="no red pushes since",
            author="aug",
        ),
    ),
    (
        "affirm",
        3,
        dict(
            key="deps:no-new-without-ask",
            instruction="Do not add dependencies without asking.",
            failure="agent pulled in `requests` for one call",
            hypothesis="agent prefers libraries to stdlib",
            outcome="dependency count flat",
            author="aug",
        ),
    ),
    (
        "affirm",
        7,
        dict(
            key="py38:no-walrus",
            instruction="Avoid the walrus operator; CI runs Python 3.8.",
            failure="SyntaxError on the 3.8 runner",
            hypothesis="agent targets newest syntax",
            outcome="3.8 job green",
            author="ci-bot",
        ),
    ),
    (
        "affirm",
        12,
        dict(
            key="style:ruff-120",
            instruction="Line length is 120; run `ruff format`.",
            failure="ruff failures in three PRs",
            hypothesis="default 88 assumed",
            outcome="ruff clean",
            author="aug",
        ),
    ),
    (
        "affirm",
        20,
        dict(
            key="api:keep-sync-wrapper",
            instruction="Keep the sync wrapper around the async client.",
            failure="notebook users broke when wrapper was removed",
            hypothesis="sync callers exist outside repo",
            outcome="wrapper restored",
            author="maya",
        ),
    ),
    # a fresh rule gets an objection: flagged, not killed
    (
        "contest",
        24,
        dict(
            key="api:keep-sync-wrapper",
            reason="wrapper doubles the surface; deprecate instead",
            author="aug",
            coherence=0.4,
        ),
    ),
    # the recurring one: tests skipped again -> re-affirm
    (
        "affirm",
        42,
        dict(
            key="tests:run-before-commit",
            instruction="Run `pytest -q` before every commit.",
            failure="red push at commit 41",
            hypothesis="instruction decayed from attention",
            outcome="re-affirmed",
            author="maya",
        ),
    ),
    (
        "affirm",
        61,
        dict(
            key="style:ruff-120",
            instruction="Line length is 120; run `ruff format`.",
            failure="ruff failure in PR #88",
            hypothesis="new contributor",
            outcome="ruff clean",
            author="aug",
        ),
    ),
    # CI dropped 3.8 at commit 90 -- nobody deletes the rule; someone contests the fossil
    (
        "contest",
        95,
        dict(key="py38:no-walrus", reason="CI dropped 3.8 at commit 90; rule is now a fossil", author="maya"),
    ),
    (
        "affirm",
        120,
        dict(
            key="tests:run-before-commit",
            instruction="Run `pytest -q` before every commit.",
            failure="red push at commit 118",
            hypothesis="new agent version",
            outcome="re-affirmed",
            author="ci-bot",
        ),
    ),
    (
        "affirm",
        131,
        dict(
            key="deps:no-new-without-ask",
            instruction="Do not add dependencies without asking.",
            failure="agent proposed `httpx`",
            hypothesis="same as before",
            outcome="declined, rule held",
            author="aug",
        ),
    ),
]


def build(until: float) -> tuple[FieldClaudeMd, list[str]]:
    """Replay history up to ``until``. Returns the field file and the append-only ratchet.

    Replaying instead of reading the past from one field matters: ``core``
    clamps age to ``max(0, now - timestamp)``, so a read at a past ``now``
    would count *future* records at full strength.
    """
    md, ratchet = FieldClaudeMd(), []
    for kind, commit, kw in HISTORY:
        if commit > until:
            break
        if kind == "affirm":
            md.affirm(commit=float(commit), **kw)
            if kw["key"] not in ratchet:
                ratchet.append(kw["key"])
        else:
            md.contest(commit=float(commit), **kw)
    return md, ratchet


def samples() -> list[tuple[float, int, int]]:
    out = []
    for t in range(0, 161, 5):
        md, ratchet = build(float(t))
        out.append((float(t), len(ratchet), len(md.lines(now=float(t)))))
    return out


def plot(samples: list[tuple[float, int, int]], path: str) -> None:
    import matplotlib.pyplot as plt

    t = [s[0] for s in samples]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.step(t, [s[1] for s in samples], where="post", color="#eb6834", lw=2, label="append-only CLAUDE.md (ratchet)")
    ax.step(t, [s[2] for s in samples], where="post", color="#2a78d6", lw=2, label="field-rendered CLAUDE.md")
    ax.set_xlabel("commit")
    ax.set_ylabel("live instructions")
    ax.set_ylim(0, 6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("instruction count over a repository's life", color="#0b0b0b")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", metavar="PNG")
    args = ap.parse_args()

    for t in (25.0, 100.0, 160.0):
        md, _ = build(t)
        print(md.render(now=t))
        print()
    md, ratchet = build(160.0)
    print(md.render_rationale(now=160.0))
    print()
    print(f"ratchet after 160 commits: {len(ratchet)} instructions, 0 deleted")
    print(f"field   after 160 commits: {len(md.lines(now=160.0))} live, {len(md.instructions)} records kept, 0 deleted")
    if args.plot:
        plot(samples(), args.plot)


if __name__ == "__main__":
    main()
