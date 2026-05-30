"""Metric specification and monitoring plan definitions.

MetricSpec describes a single metric to compute (name, type, optional
threshold).  MonitoringPlan groups specs with schema and identity metadata
and is the top-level configuration object passed to the Runner.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ayn_ml.core.data_selection import RandomSamplingConfig, WindowConfig
from ayn_ml.core.schema import DataSchema


class MetricType(str, Enum):
    """Enumeration of metric categories supported by ayn-ml.

    Attributes:
        performance: Supervised performance metrics (accuracy, F1, MSE, …).
        drift: Distribution-shift metrics (PSI, Wasserstein, MMD, …).
        statistics: Descriptive statistics (mean, std, quantile, …).
        fairness: Group-level disparity metrics (demographic parity, equalized odds, …).
        nlp_quality: Text-quality metrics (ROUGE, BLEU, METEOR, …).
        nlp_drift: Embedding- or token-level distribution shift.
        nlp_safety: Toxicity, bias, and safety classifiers.
        agent_performance: Agent task-completion and quality scores.
        agent_cost: Token usage and cost tracking for agent runs.
        recsys: Recommender-system metrics (ranking quality, diversity, bias, …).
        custom: User-defined metrics registered at runtime.
    """

    performance = "performance"
    drift = "drift"
    statistics = "statistics"
    fairness = "fairness"
    nlp_quality = "nlp_quality"
    nlp_drift = "nlp_drift"
    nlp_safety = "nlp_safety"
    agent_performance = "agent_performance"
    agent_cost = "agent_cost"
    recsys = "recsys"
    custom = "custom"


class MetricSpec(BaseModel):
    """Immutable specification for a single metric computation.

    Attributes:
        name: Registry name used to look up the metric implementation.
        metric_type: Category of this metric (see MetricType).  Defaults to
            ``None``; the runner resolves the type from the registry at
            runtime.  Set explicitly only for custom/unregistered metrics.
        feature_name: Column name for drift and statistics metrics that
            operate on a single feature.  Not required for performance metrics.
        params: Arbitrary keyword arguments forwarded to the metric's
            ``compute()`` method (e.g. ``{"average": "macro"}`` for F1,
            ``{"q": 0.95}`` for a quantile).
        threshold: Pass/fail threshold for the computed value.  Accepts:
            - ``None``: no evaluation, ``MetricResult.status`` will be ``None``.
            - ``float``: scalar bound (direction controlled by ``upper_bound``).
            - ``list[float]``: two-element ``[lo, hi]`` range; passes when
              ``lo <= value <= hi``.
        upper_bound: When ``True`` (default), the metric *passes* when
            ``value <= threshold``.  When ``False``, it passes when
            ``value >= threshold``.  Ignored when ``threshold`` is a list.
        metric_type: Category of this metric.  When ``None`` (default), the
            type is resolved at runtime from the metric registry using the
            ``name``.  Provide an explicit value only for custom metrics that
            are not yet registered.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    metric_type: MetricType | None = None
    feature_name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    threshold: float | list[float] | None = None
    upper_bound: bool = True


class MonitoringPlan(BaseModel):
    """Top-level configuration for a monitoring run.

    A MonitoringPlan binds together the data schema, model identity, the
    list of metrics to evaluate, and optional data-loading directives.
    It is the primary input to the Runner.

    Attributes:
        name: Human-readable name for this plan (used in reports and storage).
        model_id: Identifier of the model being monitored.
        model_version: Version string of the model (e.g. ``"1.2.0"``).
        data_schema: Schema describing the structure of the evaluation DataFrame.
            Accepts TabularSchema, TextSchema, AgentSchema, or RecSysSchema.
        metrics: Ordered list of metric specifications to compute.
        enable_profiling: When ``True``, the Runner computes a statistical
            profile (min, max, mean, std, percentiles, null rate, etc.) for
            every feature column referenced by the configured metrics plus the
            schema target columns (``prediction_col``, ``label_col``,
            ``proba_col``) and attaches the result to
            ``MonitoringReport.profile``.  Defaults to ``False`` — no
            profiling overhead on standard runs.
        window: How to narrow the loaded DataFrame to the current monitoring
            window.  ``None`` means the source is already pre-filtered.
        sampling: Optional random subsampling applied after window selection,
            to reduce the current window size for performance.  ``None``
            means all rows in the window are used.
        description: Optional free-text description stored alongside reports.
    """

    name: str
    model_id: str
    model_version: str
    data_schema: DataSchema
    metrics: list[MetricSpec]
    enable_profiling: bool = False
    window: WindowConfig | None = None
    sampling: RandomSamplingConfig | None = None
    description: str = ""
