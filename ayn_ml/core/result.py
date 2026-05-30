"""Runtime result types produced by metric computation and the Runner.

This module defines the output layer of ayn-ml:

- ExecutionContext  — immutable identity and time metadata for a monitoring run.
- MetricResult     — value + pass/fail status for a single metric.
- MetricError      — structured record of a metric that failed to compute.
- FiredAlert       — lightweight record of an alert that was triggered.
- MonitoringReport — top-level container aggregating all of the above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ayn_ml.core.spec import MetricSpec, MonitoringPlan


class ExecutionContext(BaseModel):
    """Immutable identity and time metadata for a single monitoring execution.

    One ExecutionContext is created per Runner invocation and attached to the
    resulting MonitoringReport.  It is NOT duplicated on individual
    MetricResult objects — retrieve it from the report instead.

    Attributes:
        run_id: Unique identifier for this run, auto-generated as a hex UUID.
            Stable within a run; use it to correlate report entries in a store.
        model_id: Identifier of the model being monitored.
        model_version: Version string of the model.
        eval_timestamp: Wall-clock time at which the run was executed (UTC).
        period_start: Start of the data window (min of timestamp_col).  ``None``
            when ``timestamp_col`` is not configured in the schema.
        period_end: End of the data window (max of timestamp_col).  ``None``
            when ``timestamp_col`` is not configured in the schema.
        n_current: Row count of the current window after all filtering.
        n_reference: Row count of the reference window, or ``None`` when no
            reference was provided.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    model_id: str
    model_version: str
    eval_timestamp: datetime
    period_start: datetime | None = None
    period_end: datetime | None = None
    n_current: int | None = None
    n_reference: int | None = None


class MetricResult(BaseModel):
    """Computed output for a single metric.

    Context (model_id, timestamp, …) lives on MonitoringReport, not here.
    MetricResult is a pure computation result: spec in → value + status out.

    Attributes:
        spec: The MetricSpec that drove this computation.
        value: Numeric or string result.  ``None`` when the metric could not
            produce a value but did not raise (edge-case only).
        status: ``True`` if the value passes the spec's threshold,
            ``False`` if it fails, or ``None`` when no threshold is defined.
        conf_interval: Optional ``(lower, upper)`` confidence interval for
            metrics that support bootstrapping (reserved for future use).
        effect_size: Standardised effect size where applicable (e.g. Cohen's d
            for t-test, Cliff's delta for Mann-Whitney, variance ratio for
            Levene, KS D-statistic for ks_2samp).  ``None`` when the metric
            does not produce an effect size.
        effect_size_label: Human-readable identifier for the effect size scale
            (e.g. ``"cohen_d"``, ``"cliff_delta"``, ``"variance_ratio"``,
            ``"ks_statistic"``).  Always set when ``effect_size`` is not
            ``None``; ``None`` otherwise.
    """

    spec: MetricSpec
    value: float | int | str | None = None
    status: bool | None = None
    conf_interval: tuple[float, float] | None = None
    effect_size: float | None = None
    effect_size_label: str | None = None


@dataclass
class MetricError:
    """Structured record of a metric that raised during computation.

    The Runner catches exceptions per-metric and converts them to MetricError
    rather than aborting the whole run.

    Attributes:
        metric_name: Registry name of the metric that failed.
        error_type: Class name of the exception (e.g. ``"SchemaError"``).
        message: Human-readable error message.
    """

    metric_name: str
    error_type: str
    message: str


@dataclass
class FiredAlert:
    """Lightweight record of an alert triggered during a monitoring run.

    FiredAlert is intentionally minimal — it carries only what is needed to
    log or notify.  Full AlertRule configuration lives in the alert layer.

    Attributes:
        metric_name: Registry name of the metric that triggered the alert.
        policy_type: Policy kind: ``"threshold"``, ``"change"``, or
            ``"consecutive"``.
        details: Policy-specific context, e.g. ``{"threshold": 0.8, "value": 0.7}``
            for a threshold alert.
        feature_name: Feature column the result belongs to, or ``None`` for
            global (plan-level) metrics such as performance and target drift.
            Populated when a single alert rule matches multiple per-feature
            results (e.g. ``psi/age`` and ``psi/income``).
    """

    metric_name: str
    policy_type: str
    details: dict[str, Any] = field(default_factory=dict)
    feature_name: str | None = None


@dataclass
class MonitoringReport:
    """Top-level output of a Runner execution.

    Aggregates all metric results, errors, and fired alerts from a single
    monitoring run, bound to a plan and execution context.

    Attributes:
        plan: The MonitoringPlan that was executed.
        context: Identity and time metadata for this run.
        results: Successful MetricResult objects, one per spec.
        errors: MetricError objects for specs that failed to compute.
        fired_alerts: Alerts triggered by AlertRule evaluation (empty when no
            alert rules are configured).
        profile: Statistical profile of feature and target columns, populated
            when ``plan.enable_profiling=True``.  ``None`` otherwise.
    """

    plan: MonitoringPlan
    context: ExecutionContext
    results: list[MetricResult]
    errors: list[MetricError]
    fired_alerts: list[FiredAlert] = field(default_factory=list)
    profile: dict[str, dict[str, float | int | str | None]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain Python dictionary.

        Nested Pydantic models are serialised with ``model_dump()``.
        Dataclass fields are converted to dicts manually to avoid a Pydantic
        dependency in the serialisation path.

        Returns:
            A JSON-serialisable dict with keys ``plan``, ``context``,
            ``results``, ``errors``, ``fired_alerts``, and (when
            ``self.profile`` is not ``None``) ``profile``.  The key
            ``profile`` is present only when ``plan.enable_profiling`` is
            ``True`` and profiling succeeded.
        """
        d = {
            "plan": self.plan.model_dump(),
            "context": self.context.model_dump(),
            "results": [r.model_dump() for r in self.results],
            "errors": [
                {
                    "metric_name": e.metric_name,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.errors
            ],
            "fired_alerts": [
                {
                    "metric_name": a.metric_name,
                    "feature_name": a.feature_name,
                    "policy_type": a.policy_type,
                    "details": a.details,
                }
                for a in self.fired_alerts
            ],
        }
        if self.profile is not None:
            d["profile"] = self.profile
        return d

    def to_dataframe(self) -> Any:
        """Convert metric results to a tidy pandas DataFrame.

        Each row represents one MetricResult.  Context fields (model_id,
        model_version, eval_timestamp) are broadcast from the report-level
        ExecutionContext rather than stored per-result.

        Returns:
            A pandas DataFrame with columns: metric_name, metric_type,
            feature_name, value, status, effect_size, effect_size_label,
            threshold, model_id, model_version, eval_timestamp.

        Raises:
            ImportError: If pandas is not installed.
        """
        import pandas as pd

        rows = []
        for r in self.results:
            rows.append(
                {
                    "metric_name": r.spec.name,
                    "metric_type": r.spec.metric_type.value if r.spec.metric_type else None,
                    "feature_name": r.spec.feature_name,
                    "value": r.value,
                    "status": r.status,
                    "effect_size": r.effect_size,
                    "effect_size_label": r.effect_size_label,
                    "threshold": r.spec.threshold,
                    "run_id": self.context.run_id,
                    "model_id": self.context.model_id,
                    "model_version": self.context.model_version,
                    "eval_timestamp": self.context.eval_timestamp,
                }
            )
        return pd.DataFrame(rows)
