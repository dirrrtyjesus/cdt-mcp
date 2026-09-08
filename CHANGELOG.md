# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **Reading the past no longer sees the future.** A write with `timestamp > now` weighs `0`
  at `now` (age was clamped to zero, so it contributed at full strength), and
  `decay_multiplier(timestamp, now)` applies only decays recorded in `[timestamp, now]`.
  `read`, `consensus`, `field`, `contested_field` and `spectral_density` with an explicit past
  `now` are now consistent with what the field looked like then. Capacity pruning ranks at a
  time no earlier than any record, and `prune()` only considers records with `timestamp <= now`,
  so a replica whose clock runs slightly ahead is neither pruned first nor dropped as "faded"
  by the next `cdt_decay` (which prunes by default).

### Added
- **`CoherenceField.compact()` / `cdt_compact`: field-conserving compaction.** Groups of write
  records sharing `(phase, payload, sign)` that have happened by `now` are replaced by one record
  stamped `now` with their summed weight. The field, the contested spectrum and every consensus
  are unchanged at every later time. Summaries carry the ids they replaced (`WriteRecord.subsumes`,
  transitive), and `absorb` honours them: originals are replaced by an arriving summary, rejected
  after one, and overlapping summaries are not double counted, so sync remains a union. Snapshot
  `schema_version` is now 3 (adds `subsumes` per record); older snapshots still load.
- `ReadResult.signed` (and `signed` on `cdt_read`): the field's component along the read
  phasor, `Re(psi[bin] * e^{-i phase})`. For a key read this is the net signed weight at the
  key. `amplitude` is `|psi|` and cannot distinguish a live proposal from one whose objections
  out-weigh it; `signed` can.

### Changed
- **Explicit decay is now an event.** `CoherenceField.decay()` records a `DecayRecord`
  (timestamp, factor) instead of rewriting each write's `scale`, and the factor is applied at
  read time to every write made at or before it. Decay events are unioned by `absorb`,
  `merge_from`, `merge` and `cdt_sync` like writes, restoring commutative/idempotent merging
  for fields that were decayed on one replica only. Snapshot `schema_version` is now 2 and
  carries a `decays` list; schema 1 snapshots still load (`scale` is folded into `value`).
- `cdt_decay` returns the recorded event's `decay_id` and `factor`; `prune_below=0` disables
  pruning so the call is purely additive.
- `WriteRecord.scale` was removed; `WriteRecord.weight()` takes the explicit multiplier as an
  argument.

### Added
- **Contested spectrum.** `consensus()` / `cdt_consensus` report `contest_ratio` (fraction of
  absolute energy lost to destructive interference) and `contested` (bins ranked by
  `Σ|w| − |Σ w·e^{iφ}|`, with payloads and agents). Perfectly opposed proposals cancel in the
  coherent field and were previously invisible to consensus. `CoherenceField.contested_field()`
  and a `contested` array in `to_dict(include_field=True)`.
- `CoherenceField.decays` / `.events`, `decay_multiplier()`, `DecayRecord` export.

### Fixed
- Consensus and read no longer recompute every impulse kernel per inspected bin
  (`O(k·N·bins)` → `O(N·bins)`).

## [0.1.0] - 2026-09-02

### Added
- `CoherenceField`: complex coherence field over phase bins with superposing writes,
  key-to-phase hashing, continuous decay, explicit decay/prune, wrapped-Gaussian kernel
  spreading, consensus extraction with payload attribution, and JSON snapshots.
- Idempotent replica synchronization: write events carry UUIDs and merging is a set union.
- `FieldStore` with atomic per-field JSON persistence.
- MCP server (`cdt-mcp`) with tools `cdt_create`, `cdt_write`, `cdt_read`, `cdt_consensus`,
  `cdt_list`, `cdt_snapshot`, `cdt_sync`, `cdt_merge`, `cdt_decay`, `cdt_delete`,
  `cdt_phase_of`; resources `cdt://fields`, `cdt://field/{name}`,
  `cdt://field/{name}/consensus`; prompt `cdt_reconcile`.
- stdio and streamable-http transports, Dockerfile, CI, test suite.
