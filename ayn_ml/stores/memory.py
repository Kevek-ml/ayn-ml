"""In-memory ResultStore backed by a collections.deque.

Intended for unit tests, integration tests, and short-lived notebook
sessions.  All data is lost when the process exits.  For persistence
across sessions use SqliteStore or another durable backend.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

from ayn_ml.core.result import MonitoringReport
from ayn_ml.stores._helpers import report_to_rows

_log = logging.getLogger(__name__)


class InMemoryStore:
    """ResultStore backed by an in-memory deque.

    Thread-safe: a single ``threading.Lock`` guards all reads and writes.
    Safe to use with the Runner in a multi-threaded context.

    Args:
        maxlen: Maximum number of ``MonitoringReport`` objects to retain.
            When the deque is full, the oldest report is evicted
            automatically on each new ``write()``.  ``None`` means
            unbounded — all reports are kept for the lifetime of the
            store instance.

    Example::

        store = InMemoryStore()
        runner = Runner()
        report = runner.run(plan, df, store=store)

        rows = store.read_history("fraud_v2", metric_name="auc")
        df = pd.DataFrame(rows)
    """

    def __init__(self, maxlen: int | None = None) -> None:
        """Initialise the store with an optional rolling-window capacity."""
        self._reports: deque[MonitoringReport] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # ResultSink
    # ------------------------------------------------------------------

    def write(self, report: MonitoringReport) -> None:
        """Append a MonitoringReport to the in-memory store.

        Args:
            report: The completed MonitoringReport from a Runner execution.
        """
        with self._lock:
            self._reports.append(report)
            _log.debug(
                "InMemoryStore: appended run_id=%s (%s %s)",
                report.context.run_id,
                report.context.model_id,
                report.context.model_version,
            )

    # ------------------------------------------------------------------
    # ResultStore
    # ------------------------------------------------------------------

    def get_report(self, run_id: str) -> MonitoringReport | None:
        """Retrieve a complete MonitoringReport by run identifier.

        Args:
            run_id: The ``ExecutionContext.run_id`` to look up.

        Returns:
            The matching ``MonitoringReport``, or ``None`` if not found.
        """
        with self._lock:
            return next(
                (r for r in self._reports if r.context.run_id == run_id),
                None,
            )

    def read_history(
        self,
        model_id: str,
        model_version: str | None = None,
        metric_name: str | None = None,
        limit: int | None = None,
        get_metadata: bool = False,
        metric_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve historical metric results as flat dictionaries.

        Iterates over stored reports newest-first, filters by the given
        criteria, and returns one dict per matching row.  Both
        MetricResult-driven rows and profile rows (when present) are
        included unless filtered by ``metric_type``.

        Args:
            model_id: Filter by model identifier.
            model_version: Optional filter by model version.
            metric_name: Optional filter by metric name (e.g. ``"auc"``).
            limit: Maximum number of rows to return.  ``None`` returns all
                matching rows.
            get_metadata: When ``True``, enrich each row with plan and run
                metadata fields (``plan_*`` and ``run_*`` prefixes).
            metric_type: Optional filter by metric type (e.g.
                ``"performance"``, ``"drift"``, ``"profile"``).  ``None``
                returns all types.

        Returns:
            List of flat dicts, newest first.  Pass directly to
            ``pd.DataFrame()`` for tabular analysis.
        """
        with self._lock:
            rows: list[dict[str, Any]] = []
            for report in reversed(self._reports):
                ctx = report.context
                if ctx.model_id != model_id:
                    continue
                if model_version is not None and ctx.model_version != model_version:
                    continue

                for row in report_to_rows(report, get_metadata=get_metadata):
                    if metric_name is not None and row["metric_name"] != metric_name:
                        continue
                    if metric_type is not None and row.get("metric_type") != metric_type:
                        continue
                    rows.append(row)
                    if limit is not None and len(rows) == limit:
                        return rows  # row-level limit reached — no over-collection

            return rows
