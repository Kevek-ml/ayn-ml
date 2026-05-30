"""Stateless orchestrator that executes a MonitoringPlan against data windows.

The Runner is the public entry point for a monitoring run.  It resolves data
sources, applies window selection and sampling, executes metrics, evaluates
alert rules, and assembles a MonitoringReport.  It is intentionally stateless
— all configuration lives in MonitoringPlan and optional runtime overrides.

Alert evaluation (ThresholdPolicy) and sink dispatch are fully implemented.
ChangePolicy and ConsecutivePolicy — which require store access — are deferred
to a future release.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    from ayn_ml.core.alert import AlertRule
    from ayn_ml.sinks.base import ResultSink

import ayn_ml.metrics  # noqa: F401 — side effect: registers all built-in metric classes
from ayn_ml.core.result import ExecutionContext, FiredAlert, MetricError, MonitoringReport
from ayn_ml.core.schema import ColumnType
from ayn_ml.core.spec import MonitoringPlan
from ayn_ml.data.sampling import (
    LastNRowsSampling,
    RandomSampling,
    TimeWindowSampling,
)
from ayn_ml.data.source import DataFrameSource, DataSource
from ayn_ml.exceptions import SchemaError
from ayn_ml.metrics.registry import get_metric
from ayn_ml.metrics.tabular._helpers import classify_columns
from ayn_ml.metrics.tabular.profiler import profile_columns

_logger = logging.getLogger(__name__)


class Runner:
    """Stateless orchestrator for a MonitoringPlan.

    Executes every MetricSpec in the plan against the supplied data windows,
    collects results and non-fatal errors, and returns a MonitoringReport.
    A single Runner instance may be reused across multiple runs.

    Args:
        n_jobs: Number of parallel workers for metric computation.
            ``1`` (default) runs metrics sequentially.  Any integer ``>= 2``
            spawns a ``ThreadPoolExecutor`` with that many workers.  ``-1``
            uses all available CPUs.  scipy and numpy release the GIL for
            most operations, so compute-heavy metrics (KS, Wasserstein, MMD,
            CBPE) benefit from thread-level parallelism.
    """

    def __init__(self, n_jobs: int = 1, strict: bool = True) -> None:
        """Initialise the Runner.

        Args:
            n_jobs: Number of parallel workers.  ``1`` = sequential.
                ``-1`` = all CPUs (passed directly to ``ThreadPoolExecutor``
                as ``max_workers=None``).  Values of ``0`` or ``< -1`` raise
                ``ValueError``.
            strict: When ``True`` (default), raise ``SchemaError`` if any
                column declared in ``data_schema`` or ``MetricSpec.feature_name``
                is absent from the DataFrame.  When ``False``, log a warning
                and continue; missing metric columns degrade to ``MetricError``.
                Use ``strict=False`` for exploratory one-shot evaluation where
                partial results are acceptable.

        Raises:
            ValueError: If ``n_jobs`` is 0 or less than -1.
        """
        if n_jobs == 0 or n_jobs < -1:
            raise ValueError(f"n_jobs must be >= 1 or -1 (all CPUs), got {n_jobs}")
        self._n_jobs = n_jobs
        self._strict = strict

    def run(
        self,
        plan: MonitoringPlan,
        current: Any,
        reference: Any | None = None,
        model: Any | None = None,
        store: Any | None = None,
        sinks: list[ResultSink] | None = None,
        alert_rules: list[AlertRule] | None = None,
    ) -> MonitoringReport:
        """Execute the monitoring plan and return a MonitoringReport.

        All data configuration (window selection, sampling) is read from
        ``plan``.  Pass data as ``current`` and optionally ``reference``;
        everything else is controlled by ``MonitoringPlan``.

        Execution order:
        1. Resolve DataSources — wrap raw DataFrames if necessary, project to
           required columns via ``DataSource.load(plan)``.
        2. Window selection — apply ``plan.window`` config.
        3. Random sub-sampling — apply ``plan.sampling`` on top of the window.
        4. Model filtering — drop rows where ``model_id_col`` or
           ``model_version_col`` do not match the plan.
        5. Context construction — build ExecutionContext with UTC timestamp
           and period bounds derived from the current window's timestamp column.
        6. Metric loop — execute each MetricSpec; per-metric errors are
           isolated and stored in ``MonitoringReport.errors``.
        7. Alert evaluation — ``ThresholdPolicy`` rules are evaluated
           against computed results; ``FiredAlert`` records are collected.
        8. Statistical profiling — compute column profiles when
           ``plan.enable_profiling`` is ``True``; skipped otherwise.
        9. Store write — ``store.write(report)`` is called when a store is
           provided; errors are logged but do not abort the run.
        10. Unconditional sink dispatch — every sink in ``sinks`` receives
            ``write(report)`` on every run, regardless of alert status.
        11. Alert channel dispatch — channels bound to fired ``AlertRule``
            objects receive ``write(report)``; errors are logged but do not
            abort the run.
        12. Return MonitoringReport.

        Args:
            plan: MonitoringPlan defining schema, metrics, and data
                configuration (window, sampling).
            current: Current-window data.  Accepts a pandas DataFrame, Polars
                DataFrame, or any ``DataSource`` implementation.
            reference: Reference-window data.  Same types as ``current``.
                May be ``None`` for metrics that do not require a reference.
            model: ModelWrapper for on-the-fly inference.  Reserved for
                Phase 5; currently ignored with a warning if supplied.
            store: ResultStore for persisting the report.  Called via
                ``store.write(report)`` unconditionally when provided.
            sinks: ResultSink list dispatched unconditionally on every run
                (e.g. a logging sink or a dashboard push).  Pass alert
                channels via ``alert_rules`` instead to limit dispatch to
                fired alerts only.
            alert_rules: AlertRule list evaluated after metric computation.
                Each rule binds a metric name to a policy and one or more
                notification channels.

        Returns:
            MonitoringReport containing all MetricResults, MetricErrors, and
            fired alerts.

        Raises:
            AynError: Re-raised only for errors outside the per-metric loop
                (e.g. DataSource resolution failures).  Per-metric errors are
                collected into the report rather than raised.
        """
        if model is not None:
            _logger.warning("Runner: 'model' argument is reserved for Phase 5 and is currently ignored.")

        # Step 1 — resolve DataSources
        current_df = self._resolve_source(current, plan)
        reference_df = self._resolve_source(reference, plan) if reference is not None else None

        # Step 2 — window selection
        current_df = self._apply_window(current_df, plan)

        # Step 3 — random sub-sampling layered on top of window
        if plan.sampling is not None:
            s = plan.sampling
            current_df = RandomSampling(n=s.n, frac=s.frac, seed=s.seed).sample(current_df, plan.data_schema)

        # Step 4 — model_id / model_version row filtering
        current_df = self._filter_model(current_df, plan)
        if reference_df is not None:
            reference_df = self._filter_model(reference_df, plan)

        # Step 5 — column validation + row counts
        n_current, n_reference = self._validate_plan_columns(current_df, reference_df, plan)

        if n_current == 0:
            _logger.warning(
                "Runner: current window is empty after model filtering "
                "(model_id=%r, model_version=%r). All metrics will likely fail.",
                plan.model_id,
                plan.model_version,
            )

        # Step 6 — ExecutionContext
        context = self._build_context(current_df, plan, n_current, n_reference)

        # Step 7 — column-kind classification (once per run) + metric loop
        column_types = classify_columns(current_df, plan.data_schema)
        results = []
        errors: list[MetricError] = []

        if self._n_jobs == 1:
            pairs = [self._run_one(spec, current_df, reference_df, plan, column_types) for spec in plan.metrics]
        else:
            max_workers = self._n_jobs if self._n_jobs != -1 else None
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_one, spec, current_df, reference_df, plan, column_types): spec
                    for spec in plan.metrics
                }
                pairs = [future.result() for future in as_completed(futures)]

        for result, error in pairs:
            if error is not None:
                _logger.warning(
                    "Metric '%s' failed with %s: %s",
                    error.metric_name,
                    error.error_type,
                    error.message,
                )
                errors.append(error)
            else:
                results.append(result)

        # Step 8 — alert evaluation (ThresholdPolicy; stateless, no store required)
        # Group by metric name — a plan may contain the same metric name for
        # multiple feature columns (e.g. psi/age, psi/income).  Each matching
        # result is evaluated independently; a rule fires once per matching result.
        fired_alerts: list[FiredAlert] = []
        if alert_rules:
            from collections import defaultdict  # noqa: PLC0415

            result_index: dict[str, list] = defaultdict(list)
            for r in results:
                result_index[r.spec.name].append(r)

            for rule in alert_rules:
                matched_results = result_index.get(rule.metric_name)
                if not matched_results:
                    _logger.warning(
                        "Runner: alert_rule references unknown metric '%s'; skipping.",
                        rule.metric_name,
                    )
                    continue
                for matched in matched_results:
                    if rule.policy.evaluate(matched):
                        alert = FiredAlert(
                            metric_name=rule.metric_name,
                            feature_name=matched.spec.feature_name,
                            policy_type=rule.policy.policy_type,
                            details=rule.policy.details(matched),
                        )
                        fired_alerts.append(alert)
                        _logger.info(
                            "Runner: alert fired — metric='%s' feature='%s' policy='%s'",
                            rule.metric_name,
                            matched.spec.feature_name,
                            rule.policy.policy_type,
                        )

        # Step 9 — statistical profiling (opt-in via plan.enable_profiling)
        col_profile: dict[str, dict[str, float | int | str | None]] | None = None
        if plan.enable_profiling:
            feature_cols: set[str] = {spec.feature_name for spec in plan.metrics if spec.feature_name}
            schema = plan.data_schema
            feature_cols.update(col for col in schema.column_names if col)
            if feature_cols:
                col_profile = profile_columns(current_df, sorted(feature_cols), plan.data_schema)

        report = MonitoringReport(
            plan=plan,
            context=context,
            results=results,
            errors=errors,
            fired_alerts=fired_alerts,
            profile=col_profile,
        )

        # Step 10 — store write
        if store is not None:
            try:
                store.write(report)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Runner: store.write() failed: %s", exc)

        # Step 11 — unconditional sink dispatch (every run, regardless of alerts)
        for sink in sinks or []:
            try:
                sink.write(report)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Runner: sink.write() failed: %s", exc)

        # Step 12 — alert channel dispatch (only when the alert fired)
        # Iterate alert_rules directly (not via a dict) so that multiple rules
        # watching the same metric_name all dispatch their channels independently.
        if fired_alerts and alert_rules:
            fired_metric_names = {fa.metric_name for fa in fired_alerts}
            for rule in alert_rules:
                if rule.metric_name not in fired_metric_names:
                    continue
                for channel in rule.channels:
                    try:
                        channel.write(report)
                    except Exception as exc:  # noqa: BLE001
                        _logger.warning(
                            "Runner: channel.write() failed for metric='%s': %s",
                            rule.metric_name,
                            exc,
                        )

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_plan_columns(
        self,
        current_df: Any,
        reference_df: Any | None,
        plan: MonitoringPlan,
    ) -> tuple[int, int | None]:
        """Validate that all columns declared in the plan exist in the DataFrames.

        Called after all filtering steps so the column set matches exactly what
        the metrics will see.  Raises ``SchemaError`` on hard failures; logs
        warnings on soft failures when ``strict=False``.

        Args:
            current_df: Current-window DataFrame after windowing, sampling, and
                model filtering.
            reference_df: Reference DataFrame, or ``None``.
            plan: MonitoringPlan providing schema and metric specs.

        Returns:
            Tuple ``(n_current, n_reference)`` where ``n_reference`` is
            ``None`` when no reference was provided.

        Raises:
            SchemaError: Always for config-level errors (``time_window`` without
                ``timestamp_col`` configured).  For runtime column absence,
                raised only when ``strict=True``.
        """
        native_current = nw.from_native(current_df, eager_only=True)
        current_cols = set(native_current.columns)
        n_current = native_current.shape[0]

        n_reference: int | None = None
        if reference_df is not None:
            native_ref = nw.from_native(reference_df, eager_only=True)
            n_reference = native_ref.shape[0]

        schema = plan.data_schema
        errors: list[str] = []

        # Infrastructure columns — declared but absent at runtime
        for attr, col in (
            ("timestamp_col", schema.timestamp_col),
            ("model_id_col", schema.model_id_col),
            ("model_version_col", schema.model_version_col),
        ):
            if col and col not in current_cols:
                msg = f"data_schema.{attr}='{col}' not found in current DataFrame"
                if self._strict:
                    errors.append(msg)
                else:
                    _logger.warning("Runner: %s", msg)

        # Metric feature_name columns
        missing_features = [
            (spec.name, spec.feature_name)
            for spec in plan.metrics
            if spec.feature_name and spec.feature_name not in current_cols
        ]
        if missing_features:
            if self._strict:
                for metric_name, col in missing_features:
                    errors.append(f"metric '{metric_name}': feature_name='{col}' not found in current DataFrame")
            else:
                cols = [col for _, col in missing_features]
                _logger.warning(
                    "Runner: %d metric(s) will likely fail — feature columns missing in current DataFrame: %s",
                    len(missing_features),
                    cols,
                )

        if errors:
            raise SchemaError("Column validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return n_current, n_reference

    def _run_one(
        self,
        spec: Any,
        current_df: Any,
        reference_df: Any | None,
        plan: MonitoringPlan,
        column_types: dict[str, ColumnType],
    ) -> tuple[Any, MetricError | None]:
        """Execute a single MetricSpec and return ``(result, None)`` or ``(None, error)``.

        Designed to be called from both sequential and parallel (thread-pool)
        contexts.  All exceptions are caught and converted to ``MetricError``
        so the caller never needs to handle them.

        Args:
            spec: MetricSpec to execute.
            current_df: Current-window DataFrame.
            reference_df: Reference-window DataFrame, or ``None``.
            plan: MonitoringPlan providing the data schema.
            column_types: Pre-computed column classification (from
                ``classify_columns``), computed once per run before the loop.

        Returns:
            A tuple ``(MetricResult, None)`` on success or
            ``(None, MetricError)`` on any failure.
        """
        try:
            metric = get_metric(spec.name)
            if metric.requires_reference and reference_df is None:
                return None, MetricError(
                    metric_name=spec.name,
                    error_type="MissingReferenceError",
                    message="Metric requires reference data but none was provided.",
                )
            # Column-kind routing — checked once here instead of per-metric extraction.
            if spec.feature_name:
                accepted = getattr(metric, "accepted_column_types", None)
                if accepted is not None:
                    col_type = column_types.get(spec.feature_name)
                    if col_type is not None and col_type not in accepted:
                        accepted_str = ", ".join(sorted(k.value for k in accepted))
                        return None, MetricError(
                            metric_name=spec.name,
                            error_type="SchemaError",
                            message=(
                                f"'{spec.feature_name}' has column type '{col_type.value}' "
                                f"but {spec.name} requires one of [{accepted_str}]. "
                                f"Declare the correct type via schema.feature_types or "
                                f"choose a compatible metric."
                            ),
                        )
            # Schema-column routing — validates columns resolved from schema attributes
            # (e.g. prediction_col, label_col) independently of spec.feature_name.
            target_types = getattr(metric, "accepted_target_types", None)
            if target_types is not None:
                target_errors = []
                for schema_attr, accepted_types in target_types.items():
                    col_name = getattr(plan.data_schema, schema_attr, None)
                    if col_name:
                        col_type = column_types.get(col_name)
                        if col_type is not None and col_type not in accepted_types:
                            accepted_str = ", ".join(sorted(k.value for k in accepted_types))
                            target_errors.append(
                                f"schema.{schema_attr}='{col_name}' has column type "
                                f"'{col_type.value}' but {spec.name} requires one of "
                                f"[{accepted_str}]"
                            )
                if target_errors:
                    return None, MetricError(
                        metric_name=spec.name,
                        error_type="SchemaError",
                        message="; ".join(target_errors) + ".",
                    )
            return metric.compute(current_df, reference_df, plan.data_schema, spec), None
        except Exception as exc:  # noqa: BLE001
            return None, MetricError(
                metric_name=spec.name,
                error_type=type(exc).__name__,
                message=str(exc),
            )

    def _resolve_source(self, data: Any, plan: MonitoringPlan) -> Any:
        """Resolve raw DataFrame or DataSource to a projected narwhals frame.

        Args:
            data: Raw DataFrame (pandas / Polars) or DataSource instance.
            plan: MonitoringPlan used to determine required columns.

        Returns:
            A narwhals-compatible frame projected to the columns required by
            the plan.
        """
        if isinstance(data, DataSource):
            return data.load(plan)
        try:
            return DataFrameSource(data).load(plan)
        except TypeError as exc:
            raise SchemaError(
                f"current/reference must be a pandas DataFrame, Polars DataFrame, "
                f"or DataSource — got {type(data).__name__}"
            ) from exc

    def _apply_window(self, df: Any, plan: MonitoringPlan) -> Any:
        """Apply window selection to the current DataFrame.

        When ``plan.window`` is ``None`` or ``"full"`` the DataFrame is
        returned unchanged.

        Args:
            df: Input DataFrame.
            plan: MonitoringPlan (provides ``plan.window`` config).

        Returns:
            Windowed DataFrame.
        """
        window = plan.window
        if window is None or window.type == "full":
            return df
        if window.type == "last_n":
            return LastNRowsSampling(window.n).sample(df, plan.data_schema)
        if window.type == "time_window":
            if not plan.data_schema.timestamp_col:
                raise SchemaError("plan.window.type='time_window' requires data_schema.timestamp_col to be configured")
            return TimeWindowSampling(window.start, window.end).sample(df, plan.data_schema)
        _logger.warning("Runner: unknown window.type=%r — returning full DataFrame.", window.type)
        return df

    def _filter_model(self, df: Any, plan: MonitoringPlan) -> Any:
        """Filter rows to match plan.model_id and plan.model_version.

        Filtering is applied only when the corresponding schema column exists
        in the DataFrame.  Missing columns are silently skipped.

        Args:
            df: Input DataFrame.
            plan: MonitoringPlan providing model_id, model_version, and schema.

        Returns:
            Filtered DataFrame (may be unchanged if no filter columns exist).
        """
        schema = plan.data_schema
        if not schema.model_id_col and not schema.model_version_col:
            return df

        native = nw.from_native(df, eager_only=True)

        if schema.model_id_col and schema.model_id_col in native.columns:
            native = native.filter(nw.col(schema.model_id_col) == plan.model_id)

        if schema.model_version_col and schema.model_version_col in native.columns:
            native = native.filter(nw.col(schema.model_version_col) == plan.model_version)

        return native.to_native()

    def _build_context(
        self,
        current_df: Any,
        plan: MonitoringPlan,
        n_current: int,
        n_reference: int | None,
    ) -> ExecutionContext:
        """Build an ExecutionContext for the current run.

        ``eval_timestamp`` is set to the current UTC time.  ``period_start``
        and ``period_end`` are derived from the min/max of the timestamp
        column when ``data_schema.timestamp_col`` is configured.

        Args:
            current_df: Current-window DataFrame after all transformations.
            plan: MonitoringPlan providing model identity and schema.
            n_current: Row count of the current window (post-filtering).
            n_reference: Row count of the reference window, or ``None``.

        Returns:
            ExecutionContext populated with model identity, timing metadata,
            and row counts.
        """
        period_start: datetime | None = None
        period_end: datetime | None = None

        schema = plan.data_schema

        if schema.timestamp_col:
            native = nw.from_native(current_df, eager_only=True)
            if schema.timestamp_col in native.columns:
                try:
                    agg = native.select(
                        nw.col(schema.timestamp_col).min().alias("_min"),
                        nw.col(schema.timestamp_col).max().alias("_max"),
                    )
                    raw_min = agg["_min"][0]
                    raw_max = agg["_max"][0]
                    # :PERF: backend scalar coercion; narwhals has no cross-backend
                    # scalar-to-stdlib-datetime utility — hasattr guard is safest available
                    period_start = raw_min.to_pydatetime() if hasattr(raw_min, "to_pydatetime") else raw_min
                    period_end = raw_max.to_pydatetime() if hasattr(raw_max, "to_pydatetime") else raw_max
                except Exception as exc:  # noqa: BLE001
                    _logger.debug(
                        "Runner: could not derive period bounds from '%s': %s",
                        schema.timestamp_col,
                        exc,
                    )

        return ExecutionContext(
            model_id=plan.model_id,
            model_version=plan.model_version,
            eval_timestamp=datetime.now(timezone.utc),
            period_start=period_start,
            period_end=period_end,
            n_current=n_current,
            n_reference=n_reference,
        )
