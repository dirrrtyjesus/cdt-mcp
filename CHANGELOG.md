# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
