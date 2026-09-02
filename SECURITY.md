# Security

## Model

- The server performs no code execution, no filesystem access outside `--state-dir`, and no
  outbound network calls.
- Persistence is JSON only. Snapshots received via `cdt_sync` are validated field by field;
  unknown keys are ignored and no deserialization of executable content occurs.
- Field names are restricted to `[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}` so they cannot escape the
  state directory.
- Per-field write events are capped (`max_records`, default 10,000) and per-store fields are
  capped (10,000) to bound memory. Payloads are capped at 64 KiB, keys at 512 chars.
- The streamable-http transport has **no authentication built in**. Bind it to localhost or put
  it behind an authenticating reverse proxy. Anyone who can reach the endpoint can read and
  write every field.

## Reporting

Please report vulnerabilities privately via GitHub Security Advisories on the repository rather
than public issues. You will get an acknowledgement within a week.
