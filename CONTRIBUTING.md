# Contributing

Thanks for helping build cdt-mcp.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Ground rules

- `src/cdt_mcp/core.py` must stay free of MCP imports. It is the library; `server.py` is the adapter.
- Every tool needs: a docstring (it becomes the tool description agents read), MCP annotations,
  a Pydantic output model, `ToolError` for user-facing failures, and a test in `tests/test_server.py`.
- Synchronization must remain idempotent, commutative and associative. Any change to
  `WriteRecord`, `absorb`, or `merge_from` needs a property test proving that in `tests/test_core.py`.
- Snapshot format changes bump `SCHEMA_VERSION` and keep `from_dict` reading older versions.
- Run `ruff check .`, `mypy src`, and `pytest` before opening a PR.

## Releasing

1. Update `CHANGELOG.md` and the version in `pyproject.toml` and `src/cdt_mcp/__init__.py`.
2. Tag `vX.Y.Z` and push. CI builds, checks, and publishes to PyPI via trusted publishing.
