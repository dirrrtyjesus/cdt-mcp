"""MCP server exposing Coherence Data Types to agents.

Run with ``cdt-mcp`` (stdio, the default) or ``cdt-mcp --transport
streamable-http --port 8000`` for multi-client deployments where several
Claude Code, Gemini, or other MCP client instances share one field store.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import __version__
from .core import (
    DEFAULT_BINS,
    DEFAULT_TAU_K,
    MAX_BINS,
    MAX_RECORDS_DEFAULT,
    BinSummary,
    CoherenceField,
    ConsensusResult,
    PayloadWeight,
    phase_from_key,
)
from .store import FieldExists, FieldNotFound, FieldStore, validate_name

logger = logging.getLogger("cdt_mcp")

INSTRUCTIONS = """\
This server hosts Coherence Data Types (CDTs): shared state that merges by
wave superposition instead of conflict resolution.

Workflow for multi-agent coordination:
1. cdt_create a named field (or let cdt_write auto-create it).
2. Each agent cdt_write its proposal: choose a `key` (any string, e.g. a
   branch name, decision label, or hypothesis) or an explicit `phase`, set
   `coherence` to your confidence (0-1), and put the human-readable proposal
   in `payload`. Negative `value` registers disagreement with that key.
3. Anyone can cdt_consensus to read the current truth: the phase bin with
   the highest spectral density, its winning payload, and alternatives.
   Check `contest_ratio` and `contested` too: opposing proposals cancel in
   the field, so a live disagreement shows up there, not in `alternatives`.
4. Replicas synchronize with cdt_snapshot -> cdt_sync. Sync is idempotent,
   so re-sending a snapshot is harmless.

Writes never overwrite. Old proposals fade only via decay (continuous
`decay_rate` on the field, or an explicit cdt_decay, which is itself a
synced event).
"""

# --------------------------------------------------------------------------- output models


class PayloadOut(BaseModel):
    payload: str | None
    weight: float
    agents: list[str]
    records: int


class FieldInfo(BaseModel):
    name: str
    bins: int
    decay_rate: float
    tau_k: float
    kernel_width: float
    max_records: int
    records: int
    spectral_density: float
    created_at: float
    updated_at: float


class CreateOut(BaseModel):
    field: FieldInfo
    created: bool


class WriteOut(BaseModel):
    field: str
    record_id: str
    phase: float
    bin: int
    key: str | None
    weight: float = Field(description="Signed weight coherence*value at write time")
    spectral_density: float


class ReadOut(BaseModel):
    field: str
    phase: float
    bin: int
    bin_phase: float
    amplitude: float
    real: float
    imag: float
    signed: float
    payloads: list[PayloadOut]


class BinOut(BaseModel):
    bin: int
    phase: float
    amplitude: float
    share: float
    payloads: list[PayloadOut]


class ConsensusOut(BaseModel):
    field: str
    phase: float
    bin: int
    amplitude: float
    confidence: float = Field(description="Peak amplitude / mean amplitude")
    share: float = Field(description="Peak amplitude / total amplitude, in [0,1]")
    spectral_density: float
    top_payload: str | None
    payloads: list[PayloadOut]
    alternatives: list[BinOut]
    record_count: int
    contest_ratio: float = Field(
        description="Fraction of total absolute energy lost to destructive interference, in [0,1]. "
        "0 = fully coherent; 1 = every proposal cancelled by an opposing one."
    )
    contested: list[BinOut] = Field(
        description="Bins ranked by contested energy sum|w| - |sum w e^(i phi)|; `amplitude` is that energy "
        "and `share` its fraction of the total. Where opposing payloads cancelled each other."
    )


class ListOut(BaseModel):
    fields: list[FieldInfo]


class SnapshotOut(BaseModel):
    snapshot: dict[str, Any]


class SyncOut(BaseModel):
    field: str
    absorbed: int = Field(description="Number of new write events absorbed")
    records: int
    consensus: ConsensusOut


class MergeOut(BaseModel):
    field: FieldInfo
    sources: list[str]
    consensus: ConsensusOut


class DecayOut(BaseModel):
    field: str
    decay_id: str = Field(description="Id of the recorded decay event (synced with the field)")
    factor: float = Field(description="Multiplier exp(-rate*dt) applied to every write made before now")
    affected: int
    pruned: int
    spectral_density: float


class DeleteOut(BaseModel):
    field: str
    deleted: bool


# --------------------------------------------------------------------------- helpers


def _payloads(pws: tuple[PayloadWeight, ...]) -> list[PayloadOut]:
    return [PayloadOut(payload=p.payload, weight=p.weight, agents=list(p.agents), records=p.records) for p in pws]


def _info(f: CoherenceField) -> FieldInfo:
    return FieldInfo(
        name=f.name,
        bins=f.bins,
        decay_rate=f.decay_rate,
        tau_k=f.tau_k,
        kernel_width=f.kernel_width,
        max_records=f.max_records,
        records=len(f),
        spectral_density=f.spectral_density(),
        created_at=f.created_at,
        updated_at=f.updated_at,
    )


def _bins(bins: tuple[BinSummary, ...]) -> list[BinOut]:
    return [
        BinOut(bin=b.bin, phase=b.phase, amplitude=b.amplitude, share=b.share, payloads=_payloads(b.payloads))
        for b in bins
    ]


def _consensus(f: CoherenceField, top_k: int = 3) -> ConsensusOut:
    c: ConsensusResult = f.consensus(top_k=top_k)
    return ConsensusOut(
        field=f.name,
        phase=c.phase,
        bin=c.bin,
        amplitude=c.amplitude,
        confidence=c.confidence,
        share=c.share,
        spectral_density=c.spectral_density,
        top_payload=c.top_payload,
        payloads=_payloads(c.payloads),
        alternatives=_bins(c.alternatives),
        record_count=c.record_count,
        contest_ratio=c.contest_ratio,
        contested=_bins(c.contested),
    )


def _get(store: FieldStore, name: str) -> CoherenceField:
    try:
        return store.get(name)
    except FieldNotFound as exc:
        raise ToolError(str(exc)) from None


READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
ADDITIVE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)


# --------------------------------------------------------------------------- server factory


def create_server(store: FieldStore | None = None, *, name: str = "cdt-mcp") -> MCPServer:
    """Build an :class:`MCPServer` bound to ``store`` (a fresh in-memory store by default)."""
    store = store if store is not None else FieldStore()
    server = MCPServer(
        name,
        title="Coherence Data Types",
        description="Multi-agent state synchronization via wave superposition (CDTs).",
        instructions=INSTRUCTIONS,
        version=__version__,
    )

    # ------------------------------------------------------------------ tools

    @server.tool(annotations=IDEMPOTENT_WRITE)
    async def cdt_create(
        name: str,
        bins: int = DEFAULT_BINS,
        decay_rate: float = 0.0,
        tau_k: float = DEFAULT_TAU_K,
        kernel_width: float = 0.0,
        max_records: int = MAX_RECORDS_DEFAULT,
    ) -> CreateOut:
        """Create a coherence field, or return the existing one with that name.

        Args:
            name: Field identifier (letters, digits, `_ . : @ -`; max 128 chars).
            bins: Phase resolution (2-4096). More bins = finer distinction between keys.
            decay_rate: Continuous decay per second; 0 keeps state forever.
            tau_k: Temporal coherence coefficient (metadata; density-averaged on merge).
            kernel_width: Impulse spread in radians; 0 writes to a single bin.
            max_records: Cap on retained write events before weakest are pruned.
        """
        if bins > MAX_BINS:
            raise ToolError(f"bins must be <= {MAX_BINS}")
        async with store.lock:
            try:
                field, created = store.get_or_create(
                    name,
                    bins=bins,
                    decay_rate=decay_rate,
                    tau_k=tau_k,
                    kernel_width=kernel_width,
                    max_records=max_records,
                )
            except ValueError as exc:
                raise ToolError(str(exc)) from None
            return CreateOut(field=_info(field), created=created)

    @server.tool(annotations=ADDITIVE)
    async def cdt_write(
        field: str,
        value: float = 1.0,
        key: str | None = None,
        phase: float | None = None,
        coherence: float = 1.0,
        payload: str | None = None,
        agent_id: str | None = None,
        auto_create: bool = True,
    ) -> WriteOut:
        """Emit a coherence impulse into a field. Superposes with existing writes; never overwrites.

        Args:
            field: Field name (auto-created with defaults if missing and `auto_create`).
            value: Magnitude of the proposal. Negative values interfere destructively
                (register disagreement) with writes at the same phase.
            key: Semantic address, hashed to a deterministic phase. Use this for
                labels like "merge-strategy:rebase". Exactly one of key/phase.
            phase: Explicit phase in radians (wrapped to [0, 2*pi)).
            coherence: Writer confidence in [0, 1]; scales the impulse.
            payload: Human-readable content of the proposal (what is being asserted).
            agent_id: Identifier of the writing agent, for attribution.
            auto_create: Create the field if it does not exist.
        """
        if key is not None and len(key) > 512:
            raise ToolError("key must be <= 512 characters")
        if payload is not None and len(payload) > 65_536:
            raise ToolError("payload must be <= 65536 characters")
        async with store.lock:
            if field not in store:
                if not auto_create:
                    raise ToolError(f"no field named {field!r}")
                try:
                    store.create(field)
                except ValueError as exc:
                    raise ToolError(str(exc)) from None
            f = store.get(field)
            try:
                rec = f.write(
                    value,
                    key=key,
                    phase=phase,
                    coherence=coherence,
                    payload=payload,
                    agent_id=agent_id,
                )
            except ValueError as exc:
                raise ToolError(str(exc)) from None
            store.flush(field)
            density = f.spectral_density()
        logger.debug("cdt_write %s phase=%.4f w=%.4f", field, rec.phase, rec.coherence * rec.value)
        return WriteOut(
            field=field,
            record_id=rec.id,
            phase=rec.phase,
            bin=f.bin_index(rec.phase),
            key=rec.key,
            weight=rec.coherence * rec.value,
            spectral_density=density,
        )

    @server.tool(annotations=READ_ONLY)
    async def cdt_read(field: str, key: str | None = None, phase: float | None = None) -> ReadOut:
        """Sample a field's amplitude at a key or phase, with the payloads that landed there."""
        async with store.lock:
            f = _get(store, field)
            try:
                r = f.read(key=key, phase=phase)
            except ValueError as exc:
                raise ToolError(str(exc)) from None
        return ReadOut(
            field=field,
            phase=r.phase,
            bin=r.bin,
            bin_phase=r.bin_phase,
            amplitude=r.amplitude,
            real=r.real,
            imag=r.imag,
            signed=r.signed,
            payloads=_payloads(r.payloads),
        )

    @server.tool(annotations=READ_ONLY)
    async def cdt_consensus(field: str, top_k: int = 3) -> ConsensusOut:
        """Read the consensus state: the phase bin of highest spectral density.

        Returns the winning bin, its aggregated payloads (the top one is the
        current truth), `confidence` (peak/mean amplitude), `share`
        (peak/total, 0-1), and the `top_k` strongest alternatives.
        """
        if not (1 <= top_k <= 64):
            raise ToolError("top_k must be in [1, 64]")
        async with store.lock:
            f = _get(store, field)
            return _consensus(f, top_k)

    @server.tool(annotations=READ_ONLY)
    async def cdt_list() -> ListOut:
        """List all fields with their sizes and energies."""
        async with store.lock:
            return ListOut(fields=[_info(f) for f in store])

    @server.tool(annotations=READ_ONLY)
    async def cdt_snapshot(field: str, include_field: bool = False) -> SnapshotOut:
        """Export a field as a JSON snapshot for transport to another replica.

        Feed the result to `cdt_sync` on the other side. Set `include_field`
        to also embed the rendered complex field ([re, im] per bin).
        """
        async with store.lock:
            f = _get(store, field)
            return SnapshotOut(snapshot=f.to_dict(include_field=include_field))

    @server.tool(annotations=IDEMPOTENT_WRITE)
    async def cdt_sync(field: str, snapshot: dict[str, Any], create: bool = True) -> SyncOut:
        """Absorb a remote snapshot into a local field (union of write events).

        Idempotent and order-independent: applying the same snapshot twice,
        or two snapshots in either order, yields the same field.
        """
        async with store.lock:
            try:
                remote = CoherenceField.from_dict(snapshot, name=field)
            except (ValueError, KeyError, TypeError) as exc:
                raise ToolError(f"invalid snapshot: {exc}") from None
            if field not in store:
                if not create:
                    raise ToolError(f"no field named {field!r}")
                try:
                    validate_name(field)
                    store.put(remote)
                except ValueError as exc:
                    raise ToolError(str(exc)) from None
                f = remote
                absorbed = len(remote)
            else:
                f = store.get(field)
                try:
                    absorbed = f.merge_from(remote)
                except ValueError as exc:
                    raise ToolError(str(exc)) from None
            store.flush(field)
            return SyncOut(field=field, absorbed=absorbed, records=len(f), consensus=_consensus(f))

    @server.tool(annotations=IDEMPOTENT_WRITE)
    async def cdt_merge(sources: list[str], into: str) -> MergeOut:
        """Superpose several fields into `into` (created if missing). Sources are left intact.

        All sources must share the same `bins`.
        """
        if not sources:
            raise ToolError("sources must not be empty")
        async with store.lock:
            fields = [_get(store, s) for s in sources]
            try:
                validate_name(into)
                if into in store:
                    target = store.get(into)
                    for f in fields:
                        if f.name != into:
                            target.merge_from(f)
                else:
                    target = CoherenceField.merge(into, fields)
                    store.put(target)
            except ValueError as exc:
                raise ToolError(str(exc)) from None
            store.flush(into)
            return MergeOut(field=_info(target), sources=sources, consensus=_consensus(target))

    @server.tool(annotations=DESTRUCTIVE)
    async def cdt_decay(field: str, dt: float, rate: float | None = None, prune_below: float = 1e-9) -> DecayOut:
        """Record a decay event exp(-rate*dt) on every impulse written so far, then prune negligible ones.

        Use to deliberately forget stale context. `rate` defaults to the field's
        own decay_rate. The decay is stored as an event and travels with
        snapshots, so replicas that sync afterwards see the same fading.
        Pruning (if `prune_below` > 0) physically drops writes and is the
        irreversible part.
        """
        async with store.lock:
            f = _get(store, field)
            n_before = len(f.decays)
            try:
                affected = f.decay(dt, rate)
            except ValueError as exc:
                raise ToolError(str(exc)) from None
            decays = f.decays
            latest = decays[-1] if len(decays) > n_before else None
            pruned = f.prune(prune_below) if prune_below > 0 else 0
            store.flush(field)
            return DecayOut(
                field=field,
                decay_id=latest.id if latest else "",
                factor=latest.factor if latest else 1.0,
                affected=affected,
                pruned=pruned,
                spectral_density=f.spectral_density(),
            )

    @server.tool(annotations=DESTRUCTIVE)
    async def cdt_delete(field: str) -> DeleteOut:
        """Delete a field and its persisted snapshot. Irreversible."""
        async with store.lock:
            deleted = store.delete(field)
        return DeleteOut(field=field, deleted=deleted)

    @server.tool(annotations=READ_ONLY)
    async def cdt_phase_of(key: str) -> dict[str, float]:
        """Return the deterministic phase (radians) a key hashes to. Pure function."""
        return {"phase": phase_from_key(key)}

    # ------------------------------------------------------------------ resources

    @server.resource("cdt://fields", name="fields", mime_type="application/json")
    async def fields_resource() -> str:
        """All fields and their summaries."""
        async with store.lock:
            return json.dumps({"fields": [_info(f).model_dump() for f in store]}, indent=2)

    @server.resource("cdt://field/{name}", name="field_snapshot", mime_type="application/json")
    async def field_resource(name: str) -> str:
        """Full JSON snapshot of one field, including the rendered complex field."""
        async with store.lock:
            try:
                f = store.get(name)
            except FieldNotFound as exc:
                raise ResourceError(str(exc)) from None
            return json.dumps(f.to_dict(include_field=True), indent=2)

    @server.resource("cdt://field/{name}/consensus", name="field_consensus", mime_type="application/json")
    async def consensus_resource(name: str) -> str:
        """Current consensus of one field."""
        async with store.lock:
            try:
                f = store.get(name)
            except FieldNotFound as exc:
                raise ResourceError(str(exc)) from None
            return _consensus(f).model_dump_json(indent=2)

    # ------------------------------------------------------------------ prompts

    @server.prompt(name="cdt_reconcile", title="Reconcile agent proposals via a CDT")
    def reconcile_prompt(field: str, topic: str) -> str:
        """Guide an agent through proposing, syncing, and reading consensus on a topic."""
        return (
            f"You are coordinating with other agents on: {topic}\n\n"
            f"Use the coherence field `{field}`.\n"
            "1. Call cdt_consensus to see what has already been proposed and how strongly.\n"
            "2. Form your own position. Call cdt_write with a stable `key` naming the option "
            "(reuse an existing key if you agree with it), `coherence` = your confidence (0-1), "
            "`payload` = a one-sentence statement of the option, and your `agent_id`.\n"
            "   To register disagreement with an option, write to its key with a negative `value`.\n"
            "3. Call cdt_consensus again. The `top_payload` at the highest-density bin is the "
            "current group truth; `share` tells you how dominant it is. Then look at "
            "`contest_ratio` and `contested`: options that were proposed and opposed cancel out "
            "of the field, so an open disagreement appears there rather than in `alternatives`.\n"
            "4. Report the consensus, its share, the strongest alternative, and any contested "
            "option with the agents on each side."
        )

    _ = (cdt_create, cdt_write, cdt_read, cdt_consensus, cdt_list, cdt_snapshot, cdt_sync)
    _ = (cdt_merge, cdt_decay, cdt_delete, cdt_phase_of, fields_resource, field_resource)
    _ = (consensus_resource, reconcile_prompt, FieldExists)
    return server


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cdt-mcp", description="Coherence Data Types MCP server")
    p.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.environ.get("CDT_MCP_TRANSPORT", "stdio"),
        help="Transport (default: stdio; env CDT_MCP_TRANSPORT)",
    )
    p.add_argument("--host", default=os.environ.get("CDT_MCP_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("CDT_MCP_PORT", "8000")))
    p.add_argument(
        "--state-dir",
        default=os.environ.get("CDT_MCP_STATE_DIR"),
        help="Directory for JSON persistence (default: in-memory only; env CDT_MCP_STATE_DIR)",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("CDT_MCP_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument("--version", action="version", version=f"cdt-mcp {__version__}")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # stdio transport owns stdout; keep logs on stderr.
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = FieldStore(args.state_dir)
    if store.state_dir:
        logger.info("loaded %d field(s) from %s", len(store), store.state_dir)
    server = create_server(store)
    transport: Literal["stdio", "streamable-http"] = args.transport
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        logger.info("serving streamable-http on http://%s:%d/mcp", args.host, args.port)
        server.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
