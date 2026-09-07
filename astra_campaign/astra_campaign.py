"""An orchestrator that routes results versus one that reads a field.

The scenario is a refactor campaign run by an orchestrator over several
workers. Each worker returns a *result*: a proposal about where test
requests should go, plus a confidence. Two orchestrators watch the same
stream of results.

``Router`` does what a last-write-wins orchestrator does: it treats each
incoming result as a local task, adopts it, and moves on. Its policy is
whichever result arrived most recently.

``FieldReader`` writes every result into one :class:`CoherenceField` and
reads consensus off spectral density. Nothing is routed; everything
superposes. Opposing results cancel in the coherent field and show up in
the *contested* spectrum instead of vanishing.

The stream is built so that the two disagree in the way the postmortem
describes -- the router's *explanations* keep improving while its
*execution* keeps flipping -- and so that the disagreement between the
workers is visible to the field reader and invisible to the router.

Run:  python astra_campaign/astra_campaign.py [--plot ring.png]

The module only needs ``cdt_mcp.core`` (and matplotlib for ``--plot``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from cdt_mcp.core import CoherenceField, ConsensusResult, phase_from_key

PARENT_QUEUE = "tests:parent-owned-queue"
BROAD_SUITES = "tests:workers-launch-broad-suites"
EXPLANATION = "explanation:bottleneck-identified"

PAYLOAD = {
    PARENT_QUEUE: "workers submit test requests to a parent-owned queue",
    BROAD_SUITES: "each worker independently launches a broad suite",
    EXPLANATION: "bottleneck recognized and explained (no execution change)",
}


@dataclass(frozen=True)
class Result:
    """What one worker hands back to the orchestrator."""

    t: float
    worker: str
    key: str
    coherence: float
    value: float = 1.0  # -1.0 is a worker arguing *against* a proposal


# One refactor session, in arrival order. Timestamps are seconds into the
# campaign. The pattern is the one from the postmortem: a bounded worker
# keeps finding the same fix; a broad-suite worker keeps reproducing the
# bottleneck; a third worker pushes back on the broad-suite approach after
# hitting the bottleneck; and the orchestrator's own explanations get
# better and better without changing anything.
STREAM: tuple[Result, ...] = (
    Result(0, "worker-1", PARENT_QUEUE, 0.80),
    Result(5, "worker-2", BROAD_SUITES, 0.60),
    Result(9, "orchestrator", EXPLANATION, 0.50),
    Result(12, "worker-3", BROAD_SUITES, 0.70, value=-1.0),  # hit the bottleneck
    Result(16, "worker-1", PARENT_QUEUE, 0.85),
    Result(20, "orchestrator", EXPLANATION, 0.75),
    Result(24, "worker-2", BROAD_SUITES, 0.65),
    Result(27, "orchestrator", EXPLANATION, 0.95),  # "explained it correctly"
    Result(30, "worker-3", BROAD_SUITES, 0.80, value=-1.0),
    Result(33, "worker-1", PARENT_QUEUE, 0.90),
)


class Router:
    """Last-write-wins. Each result is a local task; adopt it and route on."""

    def __init__(self) -> None:
        self.policy: str | None = None
        self.explanation_quality = 0.0

    def receive(self, r: Result) -> None:
        if r.key == EXPLANATION:
            self.explanation_quality = max(self.explanation_quality, r.coherence)
            return
        # A negative result is still "the latest thing the worker said":
        # the router has no way to hold it *against* an earlier result, so
        # it just adopts the subject the worker was talking about.
        self.policy = r.key


class FieldReader:
    """Superpose every result; decide by reading the field.

    With ``separate_narration=False`` the orchestrator's explanations land
    in the same field as the workers' execution results. Run it that way
    first: around t=27 the explanations, each better than the last, briefly
    *become the consensus*. The field faithfully reports that the most
    coherent energy in the campaign is the orchestrator talking about the
    bottleneck. That is the postmortem, reproduced.

    With ``separate_narration=True`` explanations go to their own field.
    The execution field is then read for policy and the narration field for
    how well the situation is understood, and the two can no longer be
    confused for each other.
    """

    def __init__(self, *, separate_narration: bool = False) -> None:
        self._now = 0.0
        clock = lambda: self._now  # noqa: E731
        self.field = CoherenceField("campaign:execution", bins=64, clock=clock)
        self.narration = self.field
        if separate_narration:
            self.narration = CoherenceField("campaign:narration", bins=64, clock=clock)

    def receive(self, r: Result) -> None:
        self._now = r.t
        target = self.narration if r.key == EXPLANATION else self.field
        target.write(
            r.value,
            key=r.key,
            coherence=r.coherence,
            agent_id=r.worker,
            payload=PAYLOAD[r.key],
            timestamp=r.t,
        )

    def consensus(self) -> ConsensusResult:
        return self.field.consensus(top_k=3)


def short(key: str | None) -> str:
    return "-" if key is None else key.split(":", 1)[1]


def run(stream: tuple[Result, ...] = STREAM) -> tuple[Router, FieldReader, FieldReader]:
    router = Router()
    mixed = FieldReader(separate_narration=False)
    split = FieldReader(separate_narration=True)
    print(f"{'t':>3}  {'worker':13s} {'result':32s} {'router adopts':22s} {'one field':22s} execution field")
    print("-" * 118)
    for r in stream:
        for o in (router, mixed, split):
            o.receive(r)
        cm, cs = mixed.consensus(), split.consensus()
        sign = "+" if r.value > 0 else "-"
        flag = "  <-- narration is the consensus" if cm.top_payload == PAYLOAD[EXPLANATION] else ""
        print(
            f"{r.t:3.0f}  {r.worker:13s} {sign}{short(r.key):31s} "
            f"{short(router.policy):22s} {short(key_of(cm)):22s} {short(key_of(cs))}"
            f"  (contested {cs.contest_ratio:.2f}){flag}"
        )
    return router, mixed, split


def key_of(c: ConsensusResult) -> str | None:
    for k, p in PAYLOAD.items():
        if p == c.top_payload:
            return k
    return None


def report(router: Router, mixed: FieldReader, split: FieldReader) -> None:
    c = split.consensus()
    n = split.narration.consensus()
    print()
    print("router (last-write-wins)")
    print(f"  final policy        : {short(router.policy)}")
    print(f"  explanation quality : {router.explanation_quality:.2f}  (improved every time it spoke)")
    print("  disagreement seen   : none -- a negative result looks like any other result")
    print()
    print("one field (execution + narration superposed)")
    print(f"  final consensus     : {short(key_of(mixed.consensus()))}")
    print("  note                : narration briefly *was* the consensus (flagged above)")
    print()
    print("execution field (narration kept in its own field)")
    print(f"  consensus           : {short(key_of(c))}")
    print(f"  share at peak       : {c.share:.2f} of coherent energy")
    print(f"  confidence          : {c.confidence:.1f}x mean bin")
    print(f"  contest ratio       : {c.contest_ratio:.2f} of impulse energy cancelled")
    for b in c.contested:
        for pw in b.payloads:
            print(
                f"  contested bin {b.bin:2d}    : {short(key_of_payload(pw.payload))}  "
                f"net {pw.weight:+.2f} from {pw.records} results by {list(pw.agents)}"
            )
    print(f"  narration field     : amplitude {n.amplitude:.2f}, {n.record_count} explanations, 0 policy")


def key_of_payload(payload: str | None) -> str | None:
    for k, p in PAYLOAD.items():
        if p == payload:
            return k
    return None


def plot(reader: FieldReader, path: str) -> None:
    import matplotlib.pyplot as plt

    f = reader.field
    psi = np.abs(f.field())
    contested = f.contested_field()
    theta = f.phase_space
    width = 2 * np.pi / f.bins
    rmax = float(max(psi.max(), contested.max())) * 1.15

    fig, ax = plt.subplots(figsize=(7, 7.4), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar(theta, psi, width=width * 0.9, bottom=0, color="#2a78d6", label="coherent |ψ|  (what consensus reads)")
    ax.bar(theta, contested, width=width * 0.9, bottom=0, color="#eb6834", label="contested Σ|w| − |Σw|  (cancelled)")
    ax.set_ylim(0, rmax)

    for key in (PARENT_QUEUE, BROAD_SUITES):
        idx = f.bin_index(phase_from_key(key))
        ax.annotate(
            short(key),
            (theta[idx], rmax * 1.04),
            ha="center",
            va="center",
            fontsize=9,
            color="#52514e",
            annotation_clip=False,
        )
    ax.text(
        0.5,
        -0.02,
        "narration lives in its own field: 3 explanations, amplitude 2.20, no policy",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="#52514e",
    )

    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines["polar"].set_color("#d8d7d2")
    ax.grid(color="#e6e5e0")
    ax.set_title("execution field after 10 worker results", color="#0b0b0b", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False, fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", metavar="PNG")
    args = ap.parse_args()
    router, mixed, split = run()
    report(router, mixed, split)
    if args.plot:
        plot(split, args.plot)
