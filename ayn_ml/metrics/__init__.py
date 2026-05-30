"""Metric registry, base protocol, and built-in metric implementations."""

# trigger registration of all built-in metrics
import ayn_ml.metrics.recsys  # noqa: F401
import ayn_ml.metrics.tabular.drift  # noqa: F401
import ayn_ml.metrics.tabular.estimation  # noqa: F401
import ayn_ml.metrics.tabular.fairness  # noqa: F401
import ayn_ml.metrics.tabular.performance  # noqa: F401
import ayn_ml.metrics.tabular.statistics  # noqa: F401
import ayn_ml.metrics.tabular.tests  # noqa: F401
from ayn_ml.metrics.base import Metric, compute_status
from ayn_ml.metrics.registry import get_metric, list_metrics, register_metric

__all__ = [
    "Metric",
    "compute_status",
    "get_metric",
    "list_metrics",
    "register_metric",
]
