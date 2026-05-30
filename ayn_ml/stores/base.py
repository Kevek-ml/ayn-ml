"""ResultStore protocol: bidirectional persistence interface.

A ResultStore extends ResultSink with a read interface.  It is the
persistence layer of ayn-ml: every completed MonitoringReport is written
to the store, and alert policies query it to retrieve historical metric
values.

Concrete implementations:
    InMemoryStore  — in-memory deque, for tests and notebooks
    SqliteStore    — local SQLite database, for development and small deployments
    JsonStore      — newline-delimited JSON file (Phase 6)
    ParquetStore   — Parquet file or directory (opt-in, Phase 6)
    SqlStore       — SQLAlchemy multi-engine (opt-in, Phase 6)
    MlflowStore    — MLflow tracking server (opt-in, Phase 6)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ayn_ml.sinks.base import ResultSink

if TYPE_CHECKING:
    from ayn_ml.core.result import MonitoringReport


class ResultStore(ResultSink, Protocol):
    """Bidirectional persistence interface for MonitoringReport objects.

    Extends ``ResultSink`` with ``read_history()`` and ``get_report()``.
    Every store can also act as a sink — pass a store as ``store=`` to
    the Runner and it handles both persistence and downstream dispatch.

    The ``write()`` method (inherited from ``ResultSink``) must be
    thread-safe: the Runner may call it from a thread pool in future
    parallel execution modes.
    """

    def read_history(
        self,
        model_id: str,
        model_version: str | None = None,
        metric_name: str | None = None,
        limit: int | None = None,
        get_metadata: bool = False,
        # arch-decision 2026-05-26: additive optional param, default None,
        # backward-compatible — existing implementors and callers unaffected.
        metric_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve historical metric results as flat dictionaries.

        Each returned dict always contains the following keys:

        ``run_id``, ``model_id``, ``model_version``, ``metric_name``,
        ``feature_name``, ``value``, ``status``, ``effect_size``,
        ``effect_size_label``, ``period_start``, ``period_end``,
        ``metric_type``.

        The ``metric_type`` column discriminates row kinds:

        - ``None`` or a ``MetricType`` value (``"performance"``,
          ``"drift"``, etc.) — rows from ``MetricSpec``-driven computation.
        - ``"profile"`` — statistical profile rows produced when
          ``enable_profiling=True``; ``status`` and ``effect_size`` are
          always ``None`` for these rows.

        When ``get_metadata=True``, the dict is enriched with plan-level
        fields (prefixed ``plan_``) and run-level fields (prefixed
        ``run_``):

        ``plan_name``, ``plan_window_type``, ``plan_window_n``,
        ``plan_sampling_type``, ``plan_sampling_frac``,
        ``run_n_current``, ``run_n_reference``.

        Results are ordered by ``period_start`` descending (newest first).
        The list is directly convertible to a pandas DataFrame via
        ``pd.DataFrame(store.read_history(...))``.

        Args:
            model_id: Filter by model identifier.
            model_version: Optional filter by model version.
            metric_name: Optional filter by metric name (e.g. ``"auc"``).
            limit: Maximum number of rows to return.  ``None`` means no
                limit — the caller receives the full history.
            get_metadata: When ``True``, JOIN with the run table and
                include plan and run metadata fields in each dict.
            metric_type: Optional filter by metric type (e.g.
                ``"performance"``, ``"drift"``, ``"profile"``).  ``None``
                returns all types.

        Returns:
            List of flat dicts, newest first.
        """
        ...

    def get_report(self, run_id: str) -> MonitoringReport | None:
        """Retrieve a complete MonitoringReport by run identifier.

        Args:
            run_id: The ``ExecutionContext.run_id`` of the target run.

        Returns:
            The full ``MonitoringReport`` if found, ``None`` otherwise.
        """
        ...
