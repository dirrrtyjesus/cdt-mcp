# cdt-mcp

**Coherence Data Types (CDTs) as a Model Context Protocol server.**
Multi-agent state synchronization by wave superposition instead of conflict resolution.

[![CI](https://github.com/dirrrtyjesus/cdt-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/dirrrtyjesus/cdt-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cdt-mcp.svg)](https://pypi.org/project/cdt-mcp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

When several Claude Code, Gemini / Antigravity, or other MCP clients work simultaneously.. they need a place to reconcile diverging state. CRDTs do this by detecting conflicts
and imposing an order. A CDT does something different: every proposal is a coherence impulse
added to a shared complex field,

```
Ψ_state = Σ_i  w_i · ψ_i        with   ψ_i = coherence_i · value_i · e^{i·phase_i}
```

and the "truth" is simply the region of highest spectral density. Nothing is overwritten,
disagreement is real (negative impulses interfere destructively), stale proposals fade by
decay, and merging replicas is a commutative, associative, idempotent union.

## Install

```bash
pip install cdt-mcp            # or: uv tool install cdt-mcp
cdt-mcp --version
```

Requires Python 3.10+ and [`mcp>=2`](https://github.com/modelcontextprotocol/python-sdk).

## Use with Claude Code

```bash
claude mcp add cdt -- cdt-mcp --state-dir ~/.cdt-mcp
```

Or add it to `.mcp.json` in a project so every collaborator's Claude Code shares the config:

```json
{
  "mcpServers": {
    "cdt": {
      "command": "cdt-mcp",
      "args": ["--state-dir", ".cdt-state"]
    }
  }
}
```

## Use with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cdt": {
      "command": "cdt-mcp",
      "args": ["--state-dir", "/Users/you/.cdt-mcp"]
    }
  }
}
```

## Use with Gemini / Antigravity

In Gemini CLI or Google Antigravity, add CDT to your project's `.mcp.json` or Antigravity configuration:

```json
{
  "mcpServers": {
    "cdt": {
      "command": "cdt-mcp",
      "args": ["--state-dir", ".cdt-state"]
    }
  }
}
```

Or via Gemini CLI:

```bash
gemini mcp add cdt -- cdt-mcp --state-dir ~/.cdt-mcp
```

## Shared server for a swarm

Run one HTTP instance and point every agent at it. All clients then superpose into the same
fields without exchanging snapshots.

```bash
cdt-mcp --transport streamable-http --host 0.0.0.0 --port 8000 --state-dir /var/lib/cdt
# clients connect to http://host:8000/mcp
```

```bash
claude mcp add --transport http cdt http://localhost:8000/mcp
```

A `Dockerfile` is included:

```bash
docker build -t cdt-mcp .
docker run -p 8000:8000 -v cdt-state:/state cdt-mcp
```

## Tools

| Tool | Purpose |
|---|---|
| `cdt_create` | Create a field (idempotent). Choose `bins`, `decay_rate`, `kernel_width`. |
| `cdt_write` | Emit an impulse: `key` or `phase`, `coherence` (confidence), `payload` (the proposal), `agent_id`. Negative `value` = disagreement. Auto-creates the field. |
| `cdt_read` | Amplitude at a key/phase plus the payloads that landed there. |
| `cdt_consensus` | The highest-density bin: `top_payload`, `share` (0-1), `confidence`, and the strongest `alternatives`. |
| `cdt_snapshot` | Export a field as JSON for another replica. |
| `cdt_sync` | Absorb a remote snapshot. Idempotent union of write events. |
| `cdt_merge` | Superpose several local fields into one. |
| `cdt_decay` | Record a decay event on all writes so far, then prune negligible impulses. |
| `cdt_compact` | Rake: collapse same-phase/payload/sign writes into one record each. Field and consensus unchanged; sync stays a union. |
| `cdt_list`, `cdt_delete`, `cdt_phase_of` | Housekeeping and the key-to-phase hash. |

Resources: `cdt://fields`, `cdt://field/{name}`, `cdt://field/{name}/consensus`.
Prompt: `cdt_reconcile(field, topic)` walks an agent through propose / sync / read-consensus.

Every tool declares MCP annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) so hosts
can auto-approve the safe ones.

## Example: three agents reconcile a merge strategy

```
agent-1  cdt_write field=merge key=rebase       coherence=0.9 payload="Rebase onto main"
agent-2  cdt_write field=merge key=merge-commit coherence=0.7 payload="Merge commit"
agent-3  cdt_write field=merge key=rebase       coherence=0.4 payload="Rebase onto main" value=-1
anyone   cdt_consensus field=merge
```

```json
{
  "top_payload": "Merge commit",
  "share": 0.583,
  "confidence": 37.3,
  "alternatives": [
    {"payloads": [{"payload": "Merge commit", "weight": 0.7, "agents": ["agent-2"]}]},
    {"payloads": [{"payload": "Rebase onto main", "weight": 0.5, "agents": ["agent-1", "agent-3"]}]}
  ]
}
```

Agent 3's objection (weight −0.4) interfered destructively with agent 1's proposal (0.9 → 0.5),
so "Merge commit" carries the field. Nothing was deleted: `cdt_read key=rebase` still shows both
contributors.

## Library use

The core has no MCP dependency:

```python
from cdt_mcp import CoherenceField

a = CoherenceField("replica-a", bins=64, decay_rate=1 / 3600)  # fades over hours
a.write(1.0, key="hypothesis:H1", coherence=0.8, payload="H1", agent_id="alice")

b = CoherenceField("replica-b", bins=64)
b.write(1.0, key="hypothesis:H2", coherence=0.6, payload="H2", agent_id="bob")

a.merge_from(b)  # union of write events; idempotent
print(a.consensus().top_payload)  # "H1"
snapshot = a.to_dict()  # JSON-safe
```

See [`examples/multi_agent_merge.py`](examples/multi_agent_merge.py) for a runnable end-to-end
client script, [`astra_campaign/`](astra_campaign/) for a multi-agent orchestration case study
demonstrating the separation of narration and execution, and [`docs/THEORY.md`](docs/THEORY.md) for the
model, foundational design principles, and guarantees.

## Semantics and guarantees

* **Writes superpose.** `write` never replaces; the field is `Σ` over all retained impulses.
* **Sync is a set union of events** (writes and explicit decays) keyed by UUID. It is
  idempotent, commutative and associative, so replicas converge regardless of delivery order
  or duplication. The only exceptions are pruning operations: the `max_records` cap and
  `prune_below` drop the weakest impulses, which can make replicas diverge. Size fields
  accordingly (default 10,000 events).
* **Decay is continuous and clock-based.** An impulse's weight at time *t* is
  `coherence · value · exp(-decay_rate · (t - t_write))`. `cdt_decay` records an extra decay
  *event* that multiplies every write made before it; it syncs like a write, so replicas that
  decayed at different moments still agree.
* **Disagreement is visible.** Opposing proposals cancel in the coherent field, so
  `cdt_consensus` also returns `contest_ratio` and the most `contested` bins
  (`Σ|w| − |Σ w·e^{iφ}|` per bin) with the payloads and agents on each side.
* **Separation of narration and execution.** Explanations and operational proposals must occupy
  disjoint fields. Superposing explanations into an execution field allows articulate narration to
  out-accumulate worker impulses and masquerade as consensus. Policy must be read strictly from
  execution fields (see [`docs/THEORY.md`](docs/THEORY.md)).
* **Compaction conserves the field.** `cdt_compact` replaces groups of writes with one summary
  record each; `Ψ`, the contested spectrum and every consensus are identical at all later times.
  Summaries carry the ids they replaced, so replicas holding the originals converge on sync
  instead of double counting. Snapshot `schema_version` is 3; schema 1 and 2 snapshots load.
* **Keys hash to phases** via SHA-256, so the same key lands in the same bin on every replica.
  With 64 bins, distinct keys collide with probability ~1/64 per pair; raise `bins` if you use
  many keys in one field, or use explicit `phase` values.
* **Persistence is JSON**, one file per field, written atomically. No pickle.
* `tau_k` is carried as metadata and density-weighted on merge; it is not associative.

## Configuration

| Flag | Env | Default |
|---|---|---|
| `--transport stdio\|streamable-http` | `CDT_MCP_TRANSPORT` | `stdio` |
| `--host` / `--port` | `CDT_MCP_HOST` / `CDT_MCP_PORT` | `127.0.0.1` / `8000` |
| `--state-dir DIR` | `CDT_MCP_STATE_DIR` | in-memory only |
| `--log-level` | `CDT_MCP_LOG_LEVEL` | `INFO` |

## Development

```bash
git clone https://github.com/dirrrtyjesus/cdt-mcp && cd cdt-mcp
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && mypy src
```

## Provenance

CDTs originate in the [Fractal Harmonic Processing](https://github.com/dirrrtyjesus/fhp-computing)
paradigm and its Ublox / PTO prototypes, where game world state was stored as a coherence field
rather than a database row. This package composes with that primitive, makes synchronization
idempotent, and exposes it over MCP.

## Contributors

* **Ajdin Dracic** ([@dirrrtyjesus](https://github.com/dirrrtyjesus))
* **Claude** ([@claude](https://github.com/claude))
* **Gemini** ([@gemini-code-assist](https://github.com/apps/gemini-code-assist))

## License

MIT
