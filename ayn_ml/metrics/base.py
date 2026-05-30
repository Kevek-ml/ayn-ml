"""Base protocol and shared utilities for metric implementations.

All metric classes must satisfy the ``Metric`` Protocol.  The ``compute_status``
helper centralises pass/fail evaluation so individual metrics never need to
re-implement threshold logic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import DataSchema
from ayn_ml.core.spec import MetricSpec, MetricType


def compute_status(value: float, spec: MetricSpec) -> bool | None:
    """Evaluate whether a metric value passes its configured threshold.

    Args:
        value: The numeric metric value to evaluate.
        spec: MetricSpec carrying the threshold configuration.

    Returns:
        - ``None``  — no threshold configured (``spec.threshold is None``).
        - ``True``  — value is within / on the correct side of the threshold.
        - ``False`` — value violates the threshold.

    Notes:
        When ``spec.threshold`` is a two-element list ``[lo, hi]``, the value
        passes iff ``lo <= value <= hi``.  For a scalar threshold, the
        direction is controlled by ``spec.upper_bound``:

        - ``upper_bound=True``  → passes when ``value <= threshold``.
        - ``upper_bound=False`` → passes when ``value >= threshold``.
    """
    if spec.threshold is None:
        return None
    if isinstance(spec.threshold, list):
        lo, hi = spec.threshold[0], spec.threshold[1]
        return lo <= value <= hi
    return value <= spec.threshold if spec.upper_bound else value >= spec.threshold


@runtime_checkable
class Metric(Protocol):
    """Structural protocol that every metric implementation must satisfy.

    Using ``@runtime_checkable`` allows ``isinstance(obj, Metric)`` checks,
    which the Runner uses to validate metrics returned from the registry.

    Attributes:
        name: Registry key matching the ``@register_metric`` decorator name.
        metric_type: Category of the metric (see MetricType).
        requires_reference: ``True`` when ``reference`` must not be ``None``
            (drift and statistical-test metrics); ``False`` for performance
            and statistics metrics.
    """

    name: str
    metric_type: MetricType
    requires_reference: bool

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Compute the metric and return a result.

        Args:
            current: Current-window DataFrame (any narwhals-compatible frame).
            reference: Reference-window DataFrame, or ``None`` for metrics
                that do not require one.
            schema: Data schema describing column names and modality.
            spec: MetricSpec with name, params, and optional threshold.

        Returns:
            MetricResult containing the computed value and pass/fail status.

        Raises:
            SchemaError: If required columns are missing or the schema type
                is incompatible with this metric.
            InsufficientDataError: If the DataFrame has too few rows.
            MetricComputeError: For unexpected errors during computation.
        """
        ...
