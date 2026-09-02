"""cdt-mcp: Coherence Data Types as a Model Context Protocol server."""

from .core import (
    CoherenceField,
    ConsensusResult,
    PayloadWeight,
    ReadResult,
    WriteRecord,
    phase_from_key,
)
from .store import FieldExists, FieldNotFound, FieldStore

__version__ = "0.1.0"

__all__ = [
    "CoherenceField",
    "ConsensusResult",
    "FieldExists",
    "FieldNotFound",
    "FieldStore",
    "PayloadWeight",
    "ReadResult",
    "WriteRecord",
    "__version__",
    "phase_from_key",
]
