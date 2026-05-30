"""Result stores — persistence layer for MonitoringReport objects.

Public exports:
    ResultSink    — write-only protocol (from ayn_ml.sinks.base)
    ResultStore   — bidirectional protocol (write + read_history + get_report)
    InMemoryStore — in-memory deque, for tests and notebooks
    SqliteStore   — local SQLite database, zero extra dependencies
"""

from typing import Any

from ayn_ml.sinks.base import ResultSink
from ayn_ml.stores.base import ResultStore
from ayn_ml.stores.memory import InMemoryStore
from ayn_ml.stores.sqlite import SqliteStore

# Populated at import time by an installed extension; None otherwise.
S3Store: type[Any] | None = None

__all__ = [
    "ResultSink",
    "ResultStore",
    "InMemoryStore",
    "S3Store",
    "SqliteStore",
]
