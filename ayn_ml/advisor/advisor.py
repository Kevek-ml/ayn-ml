"""MetricAdvisor — automatic MonitoringPlan generation from data characteristics.

Given a ``TabularSchema`` and a sample DataFrame, ``MetricAdvisor`` analyses
each feature column (normality, skewness, class imbalance, optional variance
ratio against a reference window) and selects appropriate metrics for a
``MonitoringPlan``.

Example::

    from ayn_ml.advisor import MetricAdvisor
    from ayn_ml.core.schema import TabularSchema

    schema = TabularSchema(
        label_col="y_true",
        prediction_col="y_pred",
        proba_col="y_prob",
    )
    designer = MetricAdvisor(schema)

    result = designer.suggest(
        df,
        reference=ref_df,    # required — training baseline or historical window
        task_type="classification",
        name="fraud_monitor",
        model_id="fraud_v2",
        model_version="1.0",
    )
    plan  = result.plan      # MonitoringPlan ready for Runner
    warns = result.warnings  # list[str] — advisor reasoning
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import narwhals as nw

from ayn_ml.advisor._analysis import analyze_columns
from ayn_ml.advisor._plan import SuggestedPlan
from ayn_ml.advisor._rules import (
    _compute_imbalance,
    suggest_drift_specs,
    suggest_performance_specs,
)
from ayn_ml.core.spec import MetricSpec, MonitoringPlan

if TYPE_CHECKING:
    from ayn_ml.advisor._analysis import ColumnAnalysis
    from ayn_ml.core.schema import TabularSchema

_log = logging.getLogger(__name__)

_VALID_TASK_TYPES = frozenset({"classification", "regression"})


class MetricAdvisor:
    """Automatic ``MonitoringPlan`` generator from data characteristics.

    Analyses each feature column in a DataFrame (column type, sample size,
    normality, skewness, optional variance ratio vs. reference) and builds
    a ``MonitoringPlan`` with appropriate drift and performance metrics.

    The constructor takes the schema so it can be reused across multiple
    ``suggest()`` calls (e.g. different time windows) without re-specifying
    it each time.

    Args:
        schema: ``TabularSchema`` describing label, prediction, probability,
            and feature columns.

    Example::

        designer = MetricAdvisor(schema)
        result = designer.suggest(df, reference=ref_df, task_type="classification")
    """

    def __init__(self, schema: TabularSchema) -> None:
        """Initialise the designer with a fixed schema.

        Args:
            schema: ``TabularSchema`` for the dataset.  Reused across all
                ``suggest()`` calls on this instance.
        """
        self._schema = schema

    # ------------------------------------------------------------------
    # Protected hooks — override in subclasses to change routing rules
    # ------------------------------------------------------------------

    def _compute_imbalance_ratio(self, nw_df: Any, task_type: str) -> float:
        """Return the class imbalance ratio for the current window.

        Override in subclasses to replace the default majority/minority
        class ratio computation (e.g. for multi-class or cost-sensitive
        strategies).

        Args:
            nw_df: Current-window DataFrame already wrapped with narwhals.
            task_type: ``"classification"`` or ``"regression"``.

        Returns:
            Imbalance ratio (majority / minority class count).  Returns
            ``1.0`` for regression tasks.
        """
        return _compute_imbalance(nw_df, self._schema, task_type)

    def _suggest_drift_specs(
        self, analysis: ColumnAnalysis
    ) -> tuple[list[MetricSpec], list[str]]:
        """Return drift specs and warnings for one feature column.

        Override in subclasses to apply different routing rules.

        .. note::
            If your override returns specs whose ``name`` is **not** registered
            in the ayn-ml metric registry, set ``metric_type`` explicitly on
            each such spec.  Otherwise ``suggest()`` will raise
            ``UnknownMetricError`` when resolving types after all specs are
            assembled.

        Args:
            analysis: Statistical summary for the column.

        Returns:
            Tuple of ``(specs, warnings)``.
        """
        return suggest_drift_specs(analysis)

    def _suggest_performance_specs(
        self, task_type: str, imbalance_ratio: float
    ) -> tuple[list[MetricSpec], list[str]]:
        """Return performance specs and warnings.

        Override in subclasses to apply different routing rules.

        .. note::
            If your override returns specs whose ``name`` is **not** registered
            in the ayn-ml metric registry, set ``metric_type`` explicitly on
            each such spec.  Otherwise ``suggest()`` will raise
            ``UnknownMetricError`` when resolving types after all specs are
            assembled.

        Args:
            task_type: ``"classification"`` or ``"regression"``.
            imbalance_ratio: Pre-computed class imbalance ratio (from
                ``_compute_imbalance_ratio``).

        Returns:
            Tuple of ``(specs, warnings)``.
        """
        return suggest_performance_specs(self._schema, task_type, imbalance_ratio)

    def _suggest_statistics_specs(
        self, analyses: list[ColumnAnalysis]
    ) -> list[MetricSpec]:
        """Return descriptive statistics specs.

        Returns an empty list in the base class (minimal tier).
        Override in subclasses to add statistics metrics.

        .. note::
            If your override returns specs whose ``name`` is **not** registered
            in the ayn-ml metric registry, set ``metric_type`` explicitly on
            each such spec.  Otherwise ``suggest()`` will raise
            ``UnknownMetricError`` when resolving types after all specs are
            assembled.

        Args:
            analyses: Column analyses from ``analyze_columns()``.

        Returns:
            List of MetricSpec objects.
        """
        return []

    def suggest(
        self,
        df: Any,
        *,
        reference: Any,
        task_type: str = "classification",
        name: str = "suggested_plan",
        model_id: str = "",
        model_version: str = "",
    ) -> SuggestedPlan:
        """Analyse *df* and generate a ``MonitoringPlan``.

        The generated plan contains:

        - **Performance specs** — selected based on ``task_type`` and the
          class imbalance ratio of ``df``.
        - **Target drift** — monitors label distribution shift over time.
        - **Drift specs** — one set per feature column, routing to
          parametric (ttest) or non-parametric (mannwhitney) tests based on
          normality, sample size, and (when ``reference`` is provided)
          variance ratio.

        ``reference`` is used for normality routing (skewness and normality
        tests run on the reference distribution — the stable baseline you
        are monitoring against) and for variance-ratio computation
        (Levene routing).

        Args:
            df: Current-window DataFrame (narwhals-compatible: pandas,
                Polars, or any narwhals-supported backend).
            reference: Reference DataFrame (training baseline or historical
                window).  Must use the same schema as *df*.
            task_type: ``"classification"`` (default) or ``"regression"``.
                Regression detection from a float label is ambiguous, so
                this must be declared explicitly.
            name: Human-readable name for the generated ``MonitoringPlan``.
            model_id: Model identifier embedded in the plan.
            model_version: Model version string embedded in the plan.

        Returns:
            ``SuggestedPlan`` containing the generated ``MonitoringPlan``
            and a list of advisory warning strings.

        Raises:
            ValueError: If *task_type* has an invalid value or *reference*
                is ``None``.
        """
        if reference is None:
            raise ValueError(
                "MetricAdvisor.suggest() requires a reference DataFrame. "
                "Pass the training baseline or historical window as reference=<df>."
            )

        # Ensure the metric registry is populated before any list_metrics() call
        import ayn_ml.metrics  # noqa: F401, PLC0415

        if task_type not in _VALID_TASK_TYPES:
            raise ValueError(f"task_type must be one of {sorted(_VALID_TASK_TYPES)}; got {task_type!r}")

        warnings: list[str] = []
        nw_df = nw.from_native(df, eager_only=True)

        # Warn when label looks like a float but task_type is classification
        if task_type == "classification" and self._schema.label_col in nw_df.columns:
            dtype_str = str(nw_df[self._schema.label_col].dtype).lower()
            if "float" in dtype_str:
                _log.warning(
                    "MetricAdvisor: label column '%s' has float dtype but task_type='classification'. "
                    "Set task_type='regression' if this is a regression problem.",
                    self._schema.label_col,
                )

        # --- Column analysis (reuse the already-wrapped frame) ---
        analyses = analyze_columns(nw_df, self._schema, reference)

        # --- Imbalance ratio ---
        imbalance_ratio = self._compute_imbalance_ratio(nw_df, task_type)

        # --- Performance specs ---
        perf_specs, perf_warns = self._suggest_performance_specs(task_type, imbalance_ratio)
        warnings.extend(perf_warns)

        # --- Target drift ---
        target_drift_specs: list[MetricSpec] = [MetricSpec(name="target_drift")]

        # --- Drift specs per feature column ---
        drift_specs: list[MetricSpec] = []
        for analysis in analyses:
            col_specs, col_warns = self._suggest_drift_specs(analysis)
            drift_specs.extend(col_specs)
            warnings.extend(col_warns)

        # --- Statistics specs ---
        stat_specs = self._suggest_statistics_specs(analyses)

        # Resolve metric_type from the registry for any spec that doesn't carry it explicitly.
        # The registry is guaranteed to be populated — ayn_ml.metrics was imported above.
        from ayn_ml.metrics.registry import resolve_metric_type  # noqa: PLC0415

        raw_specs = perf_specs + target_drift_specs + drift_specs + stat_specs
        all_specs = [
            s if s.metric_type is not None else s.model_copy(update={"metric_type": resolve_metric_type(s)})
            for s in raw_specs
        ]

        plan = MonitoringPlan(
            name=name,
            model_id=model_id,
            model_version=model_version,
            data_schema=self._schema,
            metrics=all_specs,
        )

        _log.debug(
            "MetricAdvisor: generated plan '%s' with %d metrics, %d warnings.",
            name,
            len(all_specs),
            len(warnings),
        )

        return SuggestedPlan(plan=plan, warnings=tuple(warnings))
