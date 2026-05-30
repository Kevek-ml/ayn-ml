"""ayn-ml: ML model monitoring for tabular, NLP, and AI agent models.

Quick start::

    from ayn_ml import MonitoringPlan, TabularSchema, MetricSpec

    plan = MonitoringPlan(
        name="churn_monitor",
        model_id="churn_v2",
        model_version="1.0",
        data_schema=TabularSchema(),
        metrics=[MetricSpec(name="accuracy")],
    )
"""

from ayn_ml.advisor import MetricAdvisor, SuggestedPlan
from ayn_ml.core import (
    AgentSchema,
    AlertPolicy,
    AlertRule,
    DataSchema,
    ExecutionContext,
    FiredAlert,
    FullWindowConfig,
    LastNRowsWindowConfig,
    MetricError,
    MetricResult,
    MetricSpec,
    MetricType,
    MonitoringPlan,
    MonitoringReport,
    PartitioningConfig,
    RandomSamplingConfig,
    TabularSchema,
    TextSchema,
    ThresholdPolicy,
    TimeBasedPartitioningConfig,
    TimeWindowConfig,
    WindowConfig,
)
from ayn_ml.exceptions import (
    AynError,
    InsufficientDataError,
    MetricComputeError,
    SchemaError,
    ThresholdError,
    UnknownMetricError,
)

__version__ = "0.1.0"


def _load_extensions() -> None:
    """Discover and load installed ayn-ml extensions (e.g. ayn-ml-pro).

    Extensions register themselves via the ``ayn_ml.extensions`` entry-point
    group.  Each entry point must be a zero-argument callable that injects
    its classes into the appropriate ``ayn_ml`` sub-modules.

    Called automatically at the end of this module, after all sub-modules
    are fully imported, to avoid circular-import issues.  A broken extension
    logs a warning and is skipped — it never kills the base package.
    """
    import logging
    from importlib.metadata import entry_points

    _log = logging.getLogger(__name__)
    for ep in entry_points(group="ayn_ml.extensions"):
        try:
            ep.load()()
        except Exception as exc:  # noqa: BLE001
            _log.warning("ayn_ml extension %r failed to load: %s", ep.name, exc)


_load_extensions()

__all__ = [
    "__version__",
    "AgentSchema",
    "MetricAdvisor",
    "SuggestedPlan",
    "AlertPolicy",
    "AlertRule",
    "AynError",
    "DataSchema",
    "ExecutionContext",
    "FiredAlert",
    "FullWindowConfig",
    "InsufficientDataError",
    "LastNRowsWindowConfig",
    "MetricComputeError",
    "MetricError",
    "MetricResult",
    "MetricSpec",
    "MetricType",
    "MonitoringPlan",
    "MonitoringReport",
    "PartitioningConfig",
    "RandomSamplingConfig",
    "SchemaError",
    "TabularSchema",
    "TextSchema",
    "ThresholdError",
    "ThresholdPolicy",
    "TimeBasedPartitioningConfig",
    "TimeWindowConfig",
    "UnknownMetricError",
    "WindowConfig",
]
