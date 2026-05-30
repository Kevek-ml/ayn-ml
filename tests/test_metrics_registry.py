import pytest

from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import UnknownMetricError
from ayn_ml.metrics.registry import _REGISTRY, get_metric, list_metrics, register_metric, resolve_metric_type


def test_list_metrics_returns_sorted():
    metrics = list_metrics()
    assert metrics == sorted(metrics)
    assert len(metrics) > 0


def test_list_metrics_contains_builtins():
    metrics = list_metrics()
    for name in ("accuracy", "f1", "psi", "wasserstein", "mean", "ks_2samp"):
        assert name in metrics


def test_get_metric_returns_instance():
    metric = get_metric("accuracy")
    assert hasattr(metric, "compute")
    assert metric.name == "accuracy"


def test_get_metric_unknown_raises():
    with pytest.raises(UnknownMetricError, match="not found"):
        get_metric("does_not_exist")


def test_register_metric_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):

        @register_metric("accuracy")
        class Duplicate:
            pass


def test_register_custom_metric():
    from ayn_ml.core.result import MetricResult
    from ayn_ml.metrics.base import compute_status

    @register_metric("_test_custom")
    class CustomMetric:
        name = "_test_custom"
        metric_type = MetricType.custom
        requires_reference = False

        def compute(self, current, reference, schema, spec) -> MetricResult:
            return MetricResult(spec=spec, value=42.0, status=compute_status(42.0, spec))

    assert "_test_custom" in list_metrics()

    s = MetricSpec(name="_test_custom", metric_type=MetricType.custom)
    result = get_metric("_test_custom").compute(None, None, None, s)
    assert result.value == 42.0

    del _REGISTRY["_test_custom"]


class TestResolveMetricType:
    def test_infers_type_from_registry(self):
        spec = MetricSpec(name="accuracy")
        assert resolve_metric_type(spec) == MetricType.performance

    def test_infers_drift_type(self):
        spec = MetricSpec(name="psi", feature_name="age")
        assert resolve_metric_type(spec) == MetricType.drift

    def test_explicit_type_returned_as_is(self):
        spec = MetricSpec(name="accuracy", metric_type=MetricType.custom)
        assert resolve_metric_type(spec) == MetricType.custom

    def test_unregistered_name_raises(self):
        spec = MetricSpec(name="does_not_exist")
        with pytest.raises(UnknownMetricError, match="does_not_exist"):
            resolve_metric_type(spec)
