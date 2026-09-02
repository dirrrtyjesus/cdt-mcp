"""Registry of named coherence fields with optional JSON persistence.

Persistence is plain JSON, one file per field, written atomically. No
pickle is used anywhere, so a state directory can be shared or inspected
safely.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .core import Clock, CoherenceField

FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")


class FieldNotFound(KeyError):
    pass


class FieldExists(ValueError):
    pass


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not FIELD_NAME_RE.match(name):
        raise ValueError(
            "field name must be 1-128 chars, start with a letter or digit, "
            "and contain only letters, digits, '_', '.', ':', '@', '-'"
        )
    return name


class FieldStore:
    """In-memory registry of :class:`CoherenceField` objects.

    Args:
        state_dir: If given, every mutation is flushed to
            ``state_dir/<name>.json`` and existing files are loaded on
            construction.
        clock: Time source passed to every field (injectable for tests).
        max_fields: Upper bound on simultaneously registered fields.
    """

    def __init__(
        self,
        state_dir: str | os.PathLike[str] | None = None,
        *,
        clock: Clock = time.time,
        max_fields: int = 10_000,
    ) -> None:
        self._fields: dict[str, CoherenceField] = {}
        self._clock = clock
        self._max_fields = max_fields
        self._state_dir = Path(state_dir).expanduser() if state_dir else None
        self.lock = asyncio.Lock()
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    # ------------------------------------------------------------------ persistence

    @property
    def state_dir(self) -> Path | None:
        return self._state_dir

    def _path(self, name: str) -> Path:
        assert self._state_dir is not None
        return self._state_dir / f"{name}.json"

    def _load_all(self) -> None:
        assert self._state_dir is not None
        for path in sorted(self._state_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                field = CoherenceField.from_dict(data, clock=self._clock)
                validate_name(field.name)
            except (OSError, ValueError, KeyError, TypeError) as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"failed to load field snapshot {path}: {exc}") from exc
            self._fields[field.name] = field

    def flush(self, name: str) -> None:
        """Write one field to disk atomically (no-op without a state dir)."""
        if self._state_dir is None:
            return
        field = self._fields.get(name)
        path = self._path(name)
        if field is None:
            path.unlink(missing_ok=True)
            return
        payload = json.dumps(field.to_dict(include_field=False), separators=(",", ":"))
        fd, tmp = tempfile.mkstemp(dir=self._state_dir, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------ registry

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[CoherenceField]:
        return iter(self._fields.values())

    def __len__(self) -> int:
        return len(self._fields)

    def names(self) -> list[str]:
        return sorted(self._fields)

    def get(self, name: str) -> CoherenceField:
        try:
            return self._fields[name]
        except KeyError:
            raise FieldNotFound(f"no field named {name!r}") from None

    def create(self, name: str, **kwargs: Any) -> CoherenceField:
        validate_name(name)
        if name in self._fields:
            raise FieldExists(f"field {name!r} already exists")
        if len(self._fields) >= self._max_fields:
            raise ValueError(f"field limit reached ({self._max_fields})")
        field = CoherenceField(name, clock=self._clock, **kwargs)
        self._fields[name] = field
        self.flush(name)
        return field

    def get_or_create(self, name: str, **kwargs: Any) -> tuple[CoherenceField, bool]:
        if name in self._fields:
            return self._fields[name], False
        return self.create(name, **kwargs), True

    def put(self, field: CoherenceField, *, overwrite: bool = False) -> None:
        validate_name(field.name)
        if field.name in self._fields and not overwrite:
            raise FieldExists(f"field {field.name!r} already exists")
        if field.name not in self._fields and len(self._fields) >= self._max_fields:
            raise ValueError(f"field limit reached ({self._max_fields})")
        self._fields[field.name] = field
        self.flush(field.name)

    def delete(self, name: str) -> bool:
        existed = self._fields.pop(name, None) is not None
        self.flush(name)
        return existed
