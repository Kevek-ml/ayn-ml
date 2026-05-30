"""SQLite-backed ResultStore using the Python stdlib sqlite3 module.

Zero external dependencies — sqlite3 is part of the Python standard
library since Python 2.5.  Suitable for local development, CI pipelines,
and small-scale production deployments (single process, single machine).

For multi-process or multi-machine deployments use SqlStore (SQLAlchemy,
opt-in via ``pip install ayn-ml[sql]``).

Schema
------
Four tables are created on first use:

``monitoring_runs``
    One row per Runner execution.  Stores scalar metadata and the full
    MonitoringPlan as a JSON blob.  ``period_start`` / ``period_end``
    are stored here as well as on ``metric_results`` so that
    ``get_report()`` can reconstruct the full ExecutionContext without
    a secondary query.

``metric_results``
    Universal time-series table — one row per measurement.  The
    ``metric_type`` column discriminates rows:

    - ``"performance"``, ``"drift"``, ``"statistics"``, ``"fairness"``, … —
      rows produced by ``MetricSpec``-driven computation.
    - ``"profile"`` — rows produced by ``enable_profiling=True``.  Each
      row is one ``(column, stat_name)`` pair from the column profile.
      ``status`` and ``effect_size`` are always ``NULL`` for profile rows.

    Key fields are denormalized from the run (model_id, model_version,
    period_start, period_end) so that time-series queries require no JOIN.

``metric_errors``
    One row per MetricError.

``fired_alerts``
    One row per FiredAlert.

Thread safety
-------------
A single ``threading.Lock`` serialises all cursor operations.  SQLite
is opened with ``check_same_thread=False`` so the same connection object
can be reused across threads.  This is correct and safe because the lock
guarantees that only one thread accesses the cursor at a time.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ayn_ml.core.result import (
    ExecutionContext,
    FiredAlert,
    MetricError,
    MetricResult,
    MonitoringReport,
)
from ayn_ml.core.spec import MetricSpec, MonitoringPlan
from ayn_ml.stores._helpers import profile_to_rows

_log = logging.getLogger(__name__)

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS monitoring_runs (
    run_id          TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    eval_timestamp  TEXT NOT NULL,
    period_start    TEXT,
    period_end      TEXT,
    n_current       INTEGER,
    n_reference     INTEGER,
    plan_json       TEXT NOT NULL
)
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS metric_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    model_id         TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    metric_name      TEXT NOT NULL,
    feature_name     TEXT,
    value            NUMERIC,
    status           INTEGER,
    effect_size      REAL,
    effect_size_label TEXT,
    period_start     TEXT,
    period_end       TEXT,
    metric_type      TEXT,
    FOREIGN KEY (run_id) REFERENCES monitoring_runs (run_id)
)
"""

_CREATE_ERRORS = """
CREATE TABLE IF NOT EXISTS metric_errors (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    error_type   TEXT NOT NULL,
    message      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES monitoring_runs (run_id)
)
"""

_CREATE_ALERTS = """
CREATE TABLE IF NOT EXISTS fired_alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    feature_name TEXT,
    policy_type  TEXT NOT NULL,
    details_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES monitoring_runs (run_id)
)
"""

_IDX_RESULTS_MODEL = """
CREATE INDEX IF NOT EXISTS idx_results_model
ON metric_results (model_id, model_version)
"""

_IDX_RESULTS_PERIOD = """
CREATE INDEX IF NOT EXISTS idx_results_period
ON metric_results (model_id, metric_name, period_start)
"""


class SqliteStore:
    """ResultStore backed by a local SQLite database.

    Args:
        path: Path to the SQLite database file.  The file is created if
            it does not exist.  Pass ``":memory:"`` for a transient
            in-process database (useful for testing when you need SQL
            query behaviour without a file on disk).

    Example::

        store = SqliteStore("monitoring.db")
        runner = Runner()
        report = runner.run(plan, df, store=store)

        rows = store.read_history("fraud_v2", metric_name="auc")
        df = pd.DataFrame(rows)
    """

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the SQLite database at ``path``."""
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        """Create tables and indexes on first use."""
        with self._lock:
            cur = self._conn.cursor()
            for ddl in (
                _CREATE_RUNS,
                _CREATE_RESULTS,
                _CREATE_ERRORS,
                _CREATE_ALERTS,
                _IDX_RESULTS_MODEL,
                _IDX_RESULTS_PERIOD,
            ):
                cur.execute(ddl)
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Call explicitly when the store is no longer needed.  After
        ``close()``, all further method calls will raise
        ``sqlite3.ProgrammingError``.
        """
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SqliteStore:
        """Return self to support context-manager usage."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection on context-manager exit."""
        self.close()

    # ------------------------------------------------------------------
    # ResultSink
    # ------------------------------------------------------------------

    def write(self, report: MonitoringReport) -> None:
        """Persist a MonitoringReport to the SQLite database.

        Inserts one row into ``monitoring_runs``; one row per
        ``MetricResult`` into ``metric_results`` (with ``metric_type``
        from the spec); one row per ``(column, stat)`` pair from
        ``report.profile`` into ``metric_results`` (with
        ``metric_type = "profile"``); one row per ``MetricError`` into
        ``metric_errors``; and one row per ``FiredAlert`` into
        ``fired_alerts``.  All inserts are wrapped in a single transaction.

        ``run_id`` is used as the primary key.  Duplicate run IDs are
        detected by a pre-flight ``SELECT`` and silently ignored — the
        second call returns immediately without inserting any rows.

        Args:
            report: The completed MonitoringReport from a Runner execution.
        """
        ctx = report.context
        plan = report.plan

        period_start_iso = ctx.period_start.isoformat() if ctx.period_start else None
        period_end_iso = ctx.period_end.isoformat() if ctx.period_end else None

        run_row = (
            ctx.run_id,
            ctx.model_id,
            ctx.model_version,
            ctx.eval_timestamp.isoformat(),
            period_start_iso,
            period_end_iso,
            ctx.n_current,
            ctx.n_reference,
            json.dumps(plan.model_dump()),
        )

        result_rows = [
            (
                ctx.run_id,
                ctx.model_id,
                ctx.model_version,
                r.spec.name,
                r.spec.feature_name,
                r.value,
                int(r.status) if r.status is not None else None,
                r.effect_size,
                r.effect_size_label,
                period_start_iso,
                period_end_iso,
                r.spec.metric_type.value if r.spec.metric_type else None,
            )
            for r in report.results
        ]

        # Profile stats (enable_profiling=True) → same table, metric_type='profile'
        for pr in profile_to_rows(report):
            result_rows.append(
                (
                    pr["run_id"],
                    pr["model_id"],
                    pr["model_version"],
                    pr["metric_name"],
                    pr["feature_name"],
                    pr["value"],
                    None,
                    None,
                    None,  # status, effect_size, effect_size_label
                    pr["period_start"],
                    pr["period_end"],
                    "profile",
                )
            )

        error_rows = [(ctx.run_id, e.metric_name, e.error_type, e.message) for e in report.errors]

        alert_rows = [
            (ctx.run_id, a.metric_name, a.feature_name, a.policy_type, json.dumps(a.details))
            for a in report.fired_alerts
        ]

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM monitoring_runs WHERE run_id = ?", (ctx.run_id,))
            if cur.fetchone() is not None:
                _log.debug("SqliteStore: run_id=%s already persisted — skipping", ctx.run_id)
                return  # run already persisted — idempotent no-op
            cur.execute(
                "INSERT INTO monitoring_runs "
                "(run_id, model_id, model_version, eval_timestamp, "
                "period_start, period_end, n_current, n_reference, plan_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                run_row,
            )
            _log.debug("SqliteStore: persisted run_id=%s (%s %s)", ctx.run_id, ctx.model_id, ctx.model_version)
            cur.executemany(
                "INSERT INTO metric_results "
                "(run_id, model_id, model_version, metric_name, feature_name, "
                "value, status, effect_size, effect_size_label, "
                "period_start, period_end, metric_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                result_rows,
            )
            if error_rows:
                cur.executemany(
                    "INSERT INTO metric_errors (run_id, metric_name, error_type, message) VALUES (?, ?, ?, ?)",
                    error_rows,
                )
            if alert_rows:
                cur.executemany(
                    "INSERT INTO fired_alerts"
                    " (run_id, metric_name, feature_name, policy_type, details_json)"
                    " VALUES (?, ?, ?, ?, ?)",
                    alert_rows,
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # ResultStore
    # ------------------------------------------------------------------

    def get_report(self, run_id: str) -> MonitoringReport | None:
        """Retrieve a complete MonitoringReport by run identifier.

        Reconstructs the full object from four tables.

        Args:
            run_id: The ``ExecutionContext.run_id`` to look up.

        Returns:
            The reconstructed ``MonitoringReport``, or ``None`` if not found.
        """
        with self._lock:
            cur = self._conn.cursor()

            cur.execute("SELECT * FROM monitoring_runs WHERE run_id = ?", (run_id,))
            run_row = cur.fetchone()
            if run_row is None:
                return None

            cur.execute("SELECT * FROM metric_results WHERE run_id = ?", (run_id,))
            result_rows = cur.fetchall()

            cur.execute("SELECT * FROM metric_errors WHERE run_id = ?", (run_id,))
            error_rows = cur.fetchall()

            cur.execute("SELECT * FROM fired_alerts WHERE run_id = ?", (run_id,))
            alert_rows = cur.fetchall()

        plan = MonitoringPlan.model_validate(json.loads(run_row["plan_json"]))
        spec_lookup = {(s.name, s.feature_name): s for s in plan.metrics}
        ctx = _reconstruct_context(run_row)

        # Split metric_results rows into MetricSpec-driven vs profile stats
        profile_rows = [r for r in result_rows if r["metric_type"] == "profile"]
        regular_rows = [r for r in result_rows if r["metric_type"] != "profile"]
        results = [_reconstruct_result(r, spec_lookup) for r in regular_rows]

        # Reconstruct the profile dict: {col_name: {stat_name: value}}
        profile = None
        if profile_rows:
            profile = {}
            for r in profile_rows:
                profile.setdefault(r["feature_name"], {})[r["metric_name"]] = r["value"]

        errors = [
            MetricError(
                metric_name=r["metric_name"],
                error_type=r["error_type"],
                message=r["message"],
            )
            for r in error_rows
        ]
        alerts = [
            FiredAlert(
                metric_name=r["metric_name"],
                policy_type=r["policy_type"],
                details=json.loads(r["details_json"]),
                feature_name=r["feature_name"],
            )
            for r in alert_rows
        ]

        return MonitoringReport(
            plan=plan,
            context=ctx,
            results=results,
            errors=errors,
            fired_alerts=alerts,
            profile=profile,
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

        Queries ``metric_results`` directly (no JOIN when
        ``get_metadata=False``).  When ``get_metadata=True``, a LEFT JOIN
        with ``monitoring_runs`` is performed and plan fields are extracted
        and prefixed with ``plan_`` / ``run_``.

        Args:
            model_id: Filter by model identifier.
            model_version: Optional filter by model version.
            metric_name: Optional filter by metric name (e.g. ``"auc"``).
            limit: Maximum number of rows to return.  ``None`` returns all
                matching rows.
            get_metadata: When ``True``, JOIN with ``monitoring_runs`` and
                include plan and run metadata fields.
            metric_type: Optional filter by metric type (e.g.
                ``"performance"``, ``"drift"``, ``"profile"``).  ``None``
                returns all types.

        Returns:
            List of flat dicts, newest first (ordered by ``period_start``
            DESC).  Pass directly to ``pd.DataFrame()`` for analysis.
        """
        params: list[Any] = [model_id]
        where = "mr.model_id = ?"

        if model_version is not None:
            where += " AND mr.model_version = ?"
            params.append(model_version)
        if metric_name is not None:
            where += " AND mr.metric_name = ?"
            params.append(metric_name)
        if metric_type is not None:
            where += " AND mr.metric_type = ?"
            params.append(metric_type)

        if get_metadata:
            select = (
                "SELECT mr.*, r.plan_json, r.n_current, r.n_reference "
                "FROM metric_results mr "
                "LEFT JOIN monitoring_runs r ON mr.run_id = r.run_id "
                f"WHERE {where} "
                "ORDER BY mr.period_start DESC"
            )
        else:
            select = f"SELECT mr.* FROM metric_results mr WHERE {where} ORDER BY mr.period_start DESC"

        if limit is not None:
            select += " LIMIT ?"
            params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(select, params)
            sql_rows = cur.fetchall()

        return [_sql_row_to_dict(row, get_metadata) for row in sql_rows]


# ------------------------------------------------------------------
# Private reconstruction helpers
# ------------------------------------------------------------------


def _reconstruct_context(row: sqlite3.Row) -> ExecutionContext:
    """Reconstruct an ExecutionContext from a monitoring_runs row."""
    from datetime import datetime, timezone

    def _parse(val: str | None) -> datetime | None:
        if val is None:
            return None
        dt = datetime.fromisoformat(val)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    return ExecutionContext(
        run_id=row["run_id"],
        model_id=row["model_id"],
        model_version=row["model_version"],
        eval_timestamp=_parse(row["eval_timestamp"]),
        period_start=_parse(row["period_start"]),
        period_end=_parse(row["period_end"]),
        n_current=row["n_current"],
        n_reference=row["n_reference"],
    )


def _reconstruct_result(
    row: sqlite3.Row,
    spec_lookup: dict[tuple[str, str | None], MetricSpec] | None = None,
) -> MetricResult:
    """Reconstruct a MetricResult from a metric_results row.

    Args:
        row: A row from the ``metric_results`` table.
        spec_lookup: Optional mapping of ``(name, feature_name)`` to the
            full ``MetricSpec`` from the plan.  When provided, the original
            spec (including ``threshold``, ``upper_bound``, ``params``,
            ``metric_type``) is restored.  When ``None``, a minimal spec is
            constructed from the row columns only.
    """
    key = (row["metric_name"], row["feature_name"])
    spec = (spec_lookup or {}).get(key) or MetricSpec(
        name=row["metric_name"],
        feature_name=row["feature_name"],
    )
    status_raw = row["status"]
    return MetricResult(
        spec=spec,
        value=row["value"],
        status=bool(status_raw) if status_raw is not None else None,
        effect_size=row["effect_size"],
        effect_size_label=row["effect_size_label"],
    )


def _sql_row_to_dict(row: sqlite3.Row, get_metadata: bool) -> dict[str, Any]:
    """Convert a sqlite3.Row from metric_results (+ optional JOIN) to dict."""
    d: dict[str, Any] = {
        "run_id": row["run_id"],
        "model_id": row["model_id"],
        "model_version": row["model_version"],
        "metric_name": row["metric_name"],
        "feature_name": row["feature_name"],
        "value": row["value"],
        "status": bool(row["status"]) if row["status"] is not None else None,
        "effect_size": row["effect_size"],
        "effect_size_label": row["effect_size_label"],
        "period_start": row["period_start"],
        "period_end": row["period_end"],
        "metric_type": row["metric_type"],
    }
    if get_metadata:
        plan_data = json.loads(row["plan_json"])
        window = plan_data.get("window") or {}
        sampling = plan_data.get("sampling") or {}
        d.update(
            {
                "plan_name": plan_data.get("name"),
                "plan_window_type": window.get("type"),
                "plan_window_n": window.get("n"),
                "plan_sampling_type": sampling.get("type"),
                "plan_sampling_frac": sampling.get("frac"),
                "run_n_current": row["n_current"],
                "run_n_reference": row["n_reference"],
            }
        )
    return d
