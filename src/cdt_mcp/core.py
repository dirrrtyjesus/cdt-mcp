"""Coherence Data Types (CDTs): state synchronization by wave superposition.

A CDT stores state as a complex-valued *coherence field* over a ring of
phase bins instead of as a single mutable value. Every write emits a
coherence impulse ``coherence * value * exp(i * phase)`` into the field.
Writes never overwrite each other -- they superpose. Reading the field
returns amplitude at a phase, and *consensus* is the phase bin with the
highest spectral density::

    Psi_state = sum_i  w_i * psi_i

Compared with a CRDT, a CDT does not detect or order conflicts. All
proposals coexist in the field; the "truth" is wherever coherent energy
accumulates. Opposing proposals (negative ``value``) interfere
destructively, so a field can hold genuine disagreement.

Implementation notes
--------------------
* Write events are the source of truth. The field is derived from them,
  which makes synchronization a *set union of write events*: idempotent,
  commutative, and associative. Merging the same snapshot twice is a
  no-op, unlike naive field addition.
* Decay is continuous. A write's contribution is scaled by
  ``exp(-decay_rate * age)`` so stale state fades without a scheduler.
* Explicit decay is *also* an event. :meth:`CoherenceField.decay` appends a
  :class:`DecayRecord` (timestamp, factor) instead of rewriting the stored
  writes; every write older than the decay is multiplied by its factor at
  read time. Decay events are unioned exactly like write events, so
  replicas that decayed at different moments still converge. (Schema 1
  baked decay into a per-record ``scale``, which made ``merge`` depend on
  which replica held a record first.)
* Consensus reports *contested* energy alongside coherent energy. Per bin,
  ``contested = sum |w_i| - |sum w_i e^{i phi_i}|`` is the energy lost to
  destructive interference. Opposing proposals cancel in the coherent
  field and would otherwise vanish from view; the contested spectrum is
  where a field's disagreement becomes visible.
* Phase bins are ``linspace(0, 2*pi, bins, endpoint=False)`` with circular
  nearest-bin snapping. (The original prototype used ``endpoint=True``,
  which aliased bin 0 and the last bin.)

The module has no MCP dependency and can be used on its own.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

ComplexField = npt.NDArray[np.complex128]
RealField = npt.NDArray[np.float64]

TWO_PI = 2.0 * math.pi
SCHEMA_VERSION = 2
DEFAULT_BINS = 64
DEFAULT_TAU_K = 7.5
MAX_BINS = 4096
MAX_RECORDS_DEFAULT = 10_000

Clock = Callable[[], float]


def phase_from_key(key: str) -> float:
    """Map an arbitrary string key to a deterministic phase in ``[0, 2*pi)``.

    Lets agents address the field by semantic keys ("branch:feature-x")
    rather than raw radians. Uses the first 8 bytes of SHA-256 so the
    mapping is stable across processes and machines.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return fraction * TWO_PI


def normalize_phase(phase: float) -> float:
    """Wrap a phase into ``[0, 2*pi)``."""
    if not math.isfinite(phase):
        raise ValueError("phase must be finite")
    return phase % TWO_PI


def circular_distance(a: npt.ArrayLike, b: float) -> RealField:
    """Shortest angular distance between phases (radians), elementwise."""
    delta: RealField = np.abs((np.asarray(a, dtype=np.float64) - b + math.pi) % TWO_PI - math.pi)
    return delta


@dataclass(frozen=True)
class WriteRecord:
    """One coherence impulse. Immutable; identified by a UUID for idempotent sync."""

    id: str
    phase: float
    value: float
    coherence: float
    timestamp: float
    agent_id: str | None = None
    payload: str | None = None
    key: str | None = None

    def weight(self, now: float, decay_rate: float, explicit: float = 1.0) -> float:
        """Signed real weight of this impulse at time ``now``.

        ``explicit`` is the product of every :class:`DecayRecord` factor that
        applies to this write (see :meth:`CoherenceField.decay_multiplier`).
        """
        age = max(0.0, now - self.timestamp)
        return self.coherence * self.value * explicit * math.exp(-decay_rate * age)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "value": self.value,
            "coherence": self.coherence,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WriteRecord:
        # Schema 1 snapshots carried a baked-in ``scale`` from explicit decay.
        # Fold it into ``value`` so the record keeps the weight it had.
        scale = float(data.get("scale", 1.0))
        return cls(
            id=str(data["id"]),
            phase=normalize_phase(float(data["phase"])),
            value=float(data["value"]) * scale,
            coherence=float(data["coherence"]),
            timestamp=float(data["timestamp"]),
            agent_id=data.get("agent_id"),
            payload=data.get("payload"),
            key=data.get("key"),
        )


@dataclass(frozen=True)
class DecayRecord:
    """One explicit decay event: every write with ``timestamp <= timestamp``
    is multiplied by ``factor``. Immutable and unioned like a write, so
    replicas converge no matter where or in what order decays happened.
    """

    id: str
    timestamp: float
    factor: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "timestamp": self.timestamp, "factor": self.factor}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecayRecord:
        factor = float(data["factor"])
        if not (0.0 <= factor <= 1.0):
            raise ValueError("decay factor must be in [0, 1]")
        return cls(id=str(data["id"]), timestamp=float(data["timestamp"]), factor=factor)


Record = WriteRecord | DecayRecord


@dataclass(frozen=True)
class PayloadWeight:
    """A payload's accumulated (signed) weight within a phase bin."""

    payload: str | None
    weight: float
    agents: tuple[str, ...]
    records: int


@dataclass(frozen=True)
class ReadResult:
    phase: float
    bin: int
    bin_phase: float
    amplitude: float
    real: float
    imag: float
    payloads: tuple[PayloadWeight, ...]


@dataclass(frozen=True)
class BinSummary:
    bin: int
    phase: float
    amplitude: float
    share: float
    payloads: tuple[PayloadWeight, ...]


@dataclass(frozen=True)
class ConsensusResult:
    """The region of highest spectral density."""

    phase: float
    bin: int
    amplitude: float
    confidence: float
    """Peak amplitude divided by mean amplitude (>= 1 when any energy exists; 0 for an empty field)."""
    share: float
    """Peak amplitude divided by total amplitude, in ``[0, 1]``."""
    spectral_density: float
    """Total field energy ``sum |field|^2``."""
    top_payload: str | None
    payloads: tuple[PayloadWeight, ...]
    alternatives: tuple[BinSummary, ...]
    record_count: int
    contest_ratio: float = 0.0
    """Fraction of total absolute energy lost to destructive interference, in ``[0, 1]``.

    ``0`` means every impulse reinforces its bin; ``1`` means the field has
    cancelled itself completely (pure disagreement, nothing coherent left).
    """
    contested: tuple[BinSummary, ...] = ()
    """Bins ranked by contested energy ``sum|w| - |sum w e^{i phi}|``.

    ``amplitude`` is the contested energy and ``share`` its fraction of the
    total contested energy. These are where opposing proposals cancelled
    and thus do not show up among ``alternatives``.
    """


@dataclass(frozen=True)
class _Rendering:
    """Field, incoherent field and per-record contributions at one instant."""

    psi: ComplexField
    incoherent: RealField
    """Per-bin ``sum |w_i| k_i``: the amplitude the bin would have without interference."""
    weights: dict[str, float]
    kernels: dict[str, RealField]

    @property
    def contested(self) -> RealField:
        # Triangle inequality guarantees >= 0; clamp only rounding noise.
        c: RealField = np.maximum(self.incoherent - np.abs(self.psi), 0.0)
        return c


class CoherenceField:
    """A named coherence field over ``bins`` phase bins.

    Args:
        name: Identifier for the field.
        bins: Number of phase bins (resolution of the ring).
        decay_rate: Continuous decay per second applied to every impulse.
            ``0.0`` means state never fades on its own.
        tau_k: Temporal coherence coefficient carried as metadata and
            density-averaged on merge.
        kernel_width: Angular width (radians) of the impulse kernel. ``0``
            writes to the single nearest bin; larger values spread each
            write across neighbouring bins with a wrapped Gaussian so
            reads between bins interpolate.
        max_records: Upper bound on retained write events. When exceeded,
            the weakest events are pruned (this is the one operation that
            can make two replicas diverge; size fields accordingly).
        clock: Time source, injectable for deterministic tests.
    """

    def __init__(
        self,
        name: str,
        *,
        bins: int = DEFAULT_BINS,
        decay_rate: float = 0.0,
        tau_k: float = DEFAULT_TAU_K,
        kernel_width: float = 0.0,
        max_records: int = MAX_RECORDS_DEFAULT,
        clock: Clock = time.time,
        created_at: float | None = None,
    ) -> None:
        if not isinstance(bins, int) or bins < 2 or bins > MAX_BINS:
            raise ValueError(f"bins must be an integer in [2, {MAX_BINS}]")
        if decay_rate < 0 or not math.isfinite(decay_rate):
            raise ValueError("decay_rate must be a finite number >= 0")
        if kernel_width < 0 or not math.isfinite(kernel_width):
            raise ValueError("kernel_width must be a finite number >= 0")
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        self.name = name
        self.bins = bins
        self.decay_rate = float(decay_rate)
        self.tau_k = float(tau_k)
        self.kernel_width = float(kernel_width)
        self.max_records = int(max_records)
        self._clock = clock
        self.created_at = float(created_at if created_at is not None else clock())
        self.updated_at = self.created_at
        self.phase_space: RealField = np.linspace(0.0, TWO_PI, bins, endpoint=False, dtype=np.float64)
        self._records: dict[str, WriteRecord] = {}
        self._decays: dict[str, DecayRecord] = {}
        # Sorted decay timestamps and suffix products of their factors, rebuilt lazily.
        self._decay_index: tuple[list[float], list[float]] | None = None

    # ------------------------------------------------------------------ basics

    @property
    def records(self) -> tuple[WriteRecord, ...]:
        """Write events only (see :attr:`decays` and :attr:`events`)."""
        return tuple(self._records.values())

    @property
    def decays(self) -> tuple[DecayRecord, ...]:
        return tuple(self._decays.values())

    @property
    def events(self) -> tuple[Record, ...]:
        """Every event that defines this field: writes and explicit decays."""
        return (*self._records.values(), *self._decays.values())

    def __len__(self) -> int:
        return len(self._records)

    def decay_multiplier(self, timestamp: float) -> float:
        """Product of the factors of every explicit decay at or after ``timestamp``."""
        if not self._decays:
            return 1.0
        if self._decay_index is None:
            ordered = sorted(self._decays.values(), key=lambda d: (d.timestamp, d.id))
            stamps = [d.timestamp for d in ordered]
            suffix = [1.0] * (len(ordered) + 1)
            for i in range(len(ordered) - 1, -1, -1):
                suffix[i] = suffix[i + 1] * ordered[i].factor
            self._decay_index = (stamps, suffix)
        stamps, suffix = self._decay_index
        return suffix[bisect.bisect_left(stamps, timestamp)]

    def now(self) -> float:
        return float(self._clock())

    def bin_index(self, phase: float) -> int:
        """Index of the bin nearest ``phase`` on the circle."""
        phase = normalize_phase(phase)
        return int(np.argmin(circular_distance(self.phase_space, phase)))

    def _kernel(self, phase: float) -> RealField:
        """Distribution of one impulse across bins (sums to 1)."""
        k: RealField
        if self.kernel_width == 0.0:
            k = np.zeros(self.bins, dtype=np.float64)
            k[self.bin_index(phase)] = 1.0
            return k
        d = circular_distance(self.phase_space, phase)
        k = np.exp(-(d**2) / (2.0 * self.kernel_width**2))
        total = float(k.sum())
        return k / total if total > 0 else k

    # ------------------------------------------------------------------ writes

    def write(
        self,
        value: float = 1.0,
        *,
        phase: float | None = None,
        key: str | None = None,
        coherence: float = 1.0,
        agent_id: str | None = None,
        payload: str | None = None,
        timestamp: float | None = None,
        record_id: str | None = None,
    ) -> WriteRecord:
        """Emit a coherence impulse. Superposes; never overwrites.

        Exactly one of ``phase`` or ``key`` must be given. ``value`` may be
        negative to interfere destructively with existing proposals at the
        same phase. ``coherence`` is the writer's confidence in ``[0, 1]``.
        """
        if (phase is None) == (key is None):
            raise ValueError("provide exactly one of phase or key")
        if key is not None:
            phase = phase_from_key(key)
        assert phase is not None
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if not (0.0 <= coherence <= 1.0):
            raise ValueError("coherence must be in [0, 1]")
        rec = WriteRecord(
            id=record_id or uuid.uuid4().hex,
            phase=normalize_phase(phase),
            value=float(value),
            coherence=float(coherence),
            timestamp=float(timestamp if timestamp is not None else self.now()),
            agent_id=agent_id,
            payload=payload,
            key=key,
        )
        self._add_record(rec)
        return rec

    def _add_record(self, rec: Record) -> bool:
        if isinstance(rec, DecayRecord):
            if rec.id in self._decays:
                return False
            self._decays[rec.id] = rec
            self._decay_index = None
            self.updated_at = max(self.updated_at, self.now())
            return True
        if rec.id in self._records:
            return False
        self._records[rec.id] = rec
        self.updated_at = max(self.updated_at, self.now())
        if len(self._records) > self.max_records:
            self._prune_to_capacity()
        return True

    def _weight(self, rec: WriteRecord, now: float) -> float:
        return rec.weight(now, self.decay_rate, self.decay_multiplier(rec.timestamp))

    def _prune_to_capacity(self) -> None:
        now = self.now()
        ranked = sorted(
            self._records.values(),
            key=lambda r: (abs(self._weight(r, now)), r.timestamp),
        )
        for rec in ranked[: len(self._records) - self.max_records]:
            del self._records[rec.id]

    # ------------------------------------------------------------------ field

    def _render(self, now: float) -> _Rendering:
        """Compute the field once and keep every per-record term for reuse."""
        psi: ComplexField = np.zeros(self.bins, dtype=np.complex128)
        incoherent: RealField = np.zeros(self.bins, dtype=np.float64)
        weights: dict[str, float] = {}
        kernels: dict[str, RealField] = {}
        for rec in self._records.values():
            w = self._weight(rec, now)
            if w == 0.0:
                continue
            k = self._kernel(rec.phase)
            weights[rec.id] = w
            kernels[rec.id] = k
            psi += w * np.exp(1j * rec.phase) * k
            incoherent += abs(w) * k
        return _Rendering(psi=psi, incoherent=incoherent, weights=weights, kernels=kernels)

    def field(self, now: float | None = None) -> ComplexField:
        """Complex coherence field at time ``now`` (defaults to the clock)."""
        t = self.now() if now is None else now
        return self._render(t).psi

    def contested_field(self, now: float | None = None) -> RealField:
        """Per-bin energy lost to destructive interference at time ``now``."""
        t = self.now() if now is None else now
        return self._render(t).contested

    def spectral_density(self, now: float | None = None) -> float:
        psi = self.field(now)
        return float(np.sum(np.abs(psi) ** 2))

    def _bin_payloads(self, bin_idx: int, rendering: _Rendering) -> tuple[PayloadWeight, ...]:
        """Aggregate signed weights of the payloads landing in ``bin_idx``."""
        agg: dict[str | None, list[Any]] = {}
        for rec in self._records.values():
            w = rendering.weights.get(rec.id)
            if w is None:
                continue
            share = rendering.kernels[rec.id][bin_idx]
            if share <= 0.0:
                continue
            w *= share
            entry = agg.setdefault(rec.payload, [0.0, set(), 0])
            entry[0] += w
            if rec.agent_id:
                entry[1].add(rec.agent_id)
            entry[2] += 1
        out = [
            PayloadWeight(payload=p, weight=float(e[0]), agents=tuple(sorted(e[1])), records=int(e[2]))
            for p, e in agg.items()
        ]
        out.sort(key=lambda pw: abs(pw.weight), reverse=True)
        return tuple(out)

    def read(self, *, phase: float | None = None, key: str | None = None, now: float | None = None) -> ReadResult:
        """Sample the field at a phase (or key)."""
        if (phase is None) == (key is None):
            raise ValueError("provide exactly one of phase or key")
        if key is not None:
            phase = phase_from_key(key)
        assert phase is not None
        phase = normalize_phase(phase)
        t = self.now() if now is None else now
        rendering = self._render(t)
        idx = self.bin_index(phase)
        z = rendering.psi[idx]
        return ReadResult(
            phase=phase,
            bin=idx,
            bin_phase=float(self.phase_space[idx]),
            amplitude=float(abs(z)),
            real=float(z.real),
            imag=float(z.imag),
            payloads=self._bin_payloads(idx, rendering),
        )

    def consensus(self, *, top_k: int = 3, now: float | None = None) -> ConsensusResult:
        """Extract consensus: the bin of maximum spectral density.

        Also ranks the ``top_k`` most *contested* bins, where opposing
        proposals cancelled. A bin can be both empty in the coherent field
        and the most contested one; ``contest_ratio`` says how much of the
        field's total energy that kind of cancellation ate.
        """
        t = self.now() if now is None else now
        rendering = self._render(t)
        amps = np.abs(rendering.psi)
        total = float(amps.sum())
        density = float(np.sum(amps**2))
        contested = rendering.contested
        contested_total = float(contested.sum())
        incoherent_total = float(rendering.incoherent.sum())
        contest_ratio = contested_total / incoherent_total if incoherent_total > 0 else 0.0
        k = max(0, top_k)

        def summarize(indices: Iterable[Any], spectrum: RealField, denom: float) -> tuple[BinSummary, ...]:
            return tuple(
                BinSummary(
                    bin=int(i),
                    phase=float(self.phase_space[i]),
                    amplitude=float(spectrum[i]),
                    share=float(spectrum[i] / denom) if denom > 0 else 0.0,
                    payloads=self._bin_payloads(int(i), rendering),
                )
                for i in indices
                if spectrum[i] > 0.0
            )

        contested_bins = summarize(np.argsort(-contested, kind="stable")[:k], contested, contested_total)
        if total <= 0.0 or not self._records:
            return ConsensusResult(
                phase=0.0,
                bin=0,
                amplitude=0.0,
                confidence=0.0,
                share=0.0,
                spectral_density=0.0,
                top_payload=None,
                payloads=(),
                alternatives=(),
                record_count=len(self._records),
                contest_ratio=contest_ratio,
                contested=contested_bins,
            )
        order = np.argsort(-amps, kind="stable")
        best = int(order[0])
        mean = float(amps.mean())
        payloads = self._bin_payloads(best, rendering)
        return ConsensusResult(
            phase=float(self.phase_space[best]),
            bin=best,
            amplitude=float(amps[best]),
            confidence=float(amps[best] / mean) if mean > 0 else 0.0,
            share=float(amps[best] / total),
            spectral_density=density,
            top_payload=payloads[0].payload if payloads else None,
            payloads=payloads,
            alternatives=summarize(order[:k], amps, total),
            record_count=len(self._records),
            contest_ratio=contest_ratio,
            contested=contested_bins,
        )

    # ------------------------------------------------------------------ time

    def decay(
        self,
        dt: float,
        rate: float | None = None,
        *,
        timestamp: float | None = None,
        record_id: str | None = None,
    ) -> int:
        """Record an explicit decay ``exp(-rate * dt)`` of every impulse written so far.

        This is in addition to the continuous ``decay_rate``. The decay is
        stored as a :class:`DecayRecord` and applied at read time to writes
        with ``timestamp <= timestamp`` (default: now), so it syncs to other
        replicas like any other event. Returns the number of writes affected.
        """
        if dt < 0 or not math.isfinite(dt):
            raise ValueError("dt must be finite and >= 0")
        r = self.decay_rate if rate is None else rate
        if r < 0 or not math.isfinite(r):
            raise ValueError("rate must be finite and >= 0")
        factor = math.exp(-r * dt)
        if factor == 1.0:
            return 0
        ts = float(self.now() if timestamp is None else timestamp)
        self._add_record(DecayRecord(id=record_id or uuid.uuid4().hex, timestamp=ts, factor=factor))
        return sum(1 for rec in self._records.values() if rec.timestamp <= ts)

    def prune(self, epsilon: float = 1e-9, now: float | None = None) -> int:
        """Drop write records whose absolute weight fell below ``epsilon``.

        Like ``max_records`` pruning this is a non-monotonic operation: a
        pruned write can be re-absorbed from a replica that still holds it.
        """
        t = self.now() if now is None else now
        dead = [k for k, v in self._records.items() if abs(self._weight(v, t)) < epsilon]
        for k in dead:
            del self._records[k]
        if dead:
            self.updated_at = self.now()
        return len(dead)

    # ------------------------------------------------------------------ sync

    def absorb(self, records: Iterable[Record]) -> int:
        """Union another replica's events (writes and decays) into this field.

        Idempotent: records already present (by id) are ignored. Returns
        the number of new records absorbed.
        """
        added = 0
        for rec in records:
            if self._add_record(rec):
                added += 1
        return added

    def merge_from(self, other: CoherenceField) -> int:
        """Superpose ``other`` into this field (union of write and decay events)."""
        if other.bins != self.bins:
            raise ValueError(f"cannot merge fields with different bins ({self.bins} vs {other.bins})")
        before = self.spectral_density()
        added = self.absorb(other.events)
        # Density-weighted tau_k so the more energetic replica dominates.
        num = self.tau_k * before + other.tau_k * other.spectral_density()
        den = before + other.spectral_density()
        self.tau_k = float(num / den) if den > 0 else (self.tau_k + other.tau_k) / 2.0
        return added

    @classmethod
    def merge(cls, name: str, fields: Iterable[CoherenceField], *, clock: Clock = time.time) -> CoherenceField:
        """Superpose several fields into a new one. Sources are untouched."""
        fields = list(fields)
        if not fields:
            raise ValueError("merge requires at least one field")
        first = fields[0]
        merged = cls(
            name,
            bins=first.bins,
            decay_rate=first.decay_rate,
            tau_k=first.tau_k,
            kernel_width=first.kernel_width,
            max_records=max(f.max_records for f in fields),
            clock=clock,
        )
        merged.absorb(first.events)
        for f in fields[1:]:
            merged.merge_from(f)
        return merged

    # ------------------------------------------------------------------ (de)serialization

    def to_dict(self, *, include_field: bool = True, now: float | None = None) -> dict[str, Any]:
        """JSON-safe snapshot. Complex field values are ``[re, im]`` pairs."""
        t = self.now() if now is None else now
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "bins": self.bins,
            "decay_rate": self.decay_rate,
            "tau_k": self.tau_k,
            "kernel_width": self.kernel_width,
            "max_records": self.max_records,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "snapshot_at": t,
            "records": [r.to_dict() for r in self._records.values()],
            "decays": [d.to_dict() for d in self._decays.values()],
        }
        if include_field:
            rendering = self._render(t)
            data["field"] = [[float(z.real), float(z.imag)] for z in rendering.psi]
            data["contested"] = [float(c) for c in rendering.contested]
            data["spectral_density"] = float(np.sum(np.abs(rendering.psi) ** 2))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, clock: Clock = time.time, name: str | None = None) -> CoherenceField:
        version = int(data.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot schema_version {version}")
        f = cls(
            name or str(data["name"]),
            bins=int(data.get("bins", DEFAULT_BINS)),
            decay_rate=float(data.get("decay_rate", 0.0)),
            tau_k=float(data.get("tau_k", DEFAULT_TAU_K)),
            kernel_width=float(data.get("kernel_width", 0.0)),
            max_records=int(data.get("max_records", MAX_RECORDS_DEFAULT)),
            clock=clock,
            created_at=float(data["created_at"]) if "created_at" in data else None,
        )
        f.absorb(WriteRecord.from_dict(r) for r in data.get("records", []))
        f.absorb(DecayRecord.from_dict(d) for d in data.get("decays", []))
        if "updated_at" in data:
            f.updated_at = float(data["updated_at"])
        return f
