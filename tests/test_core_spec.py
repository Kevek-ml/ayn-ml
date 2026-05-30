from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ayn_ml.core.data_selection import LastNRowsWindowConfig, RandomSamplingConfig, TimeWindowConfig
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType, MonitoringPlan


class TestMetricSpec:
    def test_minimal_without_metric_type(self):
        spec = MetricSpec(name="accuracy")
        assert spec.name == "accuracy"
        assert spec.metric_type is None
        assert spec.feature_name is None
        assert spec.threshold is None
        assert spec.params == {}
        assert spec.upper_bound is True

    def test_explicit_metric_type_still_accepted(self):
        spec = MetricSpec(name="accuracy", metric_type=MetricType.performance)
        assert spec.metric_type == MetricType.performance

    def test_with_threshold(self):
        spec = MetricSpec(name="f1", threshold=0.8, upper_bound=False)
        assert spec.threshold == 0.8
        assert spec.upper_bound is False

    def test_with_feature(self):
        spec = MetricSpec(name="psi", feature_name="age", params={"n_bins": 10})
        assert spec.feature_name == "age"
        assert spec.params["n_bins"] == 10

    def test_params_default_is_empty_dict(self):
        s1 = MetricSpec(name="a")
        s2 = MetricSpec(name="b")
        s1.params["key"] = "value"
        assert "key" not in s2.params

    def test_frozen(self):
        spec = MetricSpec(name="accuracy")
        with pytest.raises(ValidationError):
            spec.name = "other"

    def test_list_threshold(self):
        spec = MetricSpec(name="quantile", threshold=[0.25, 0.75])
        assert spec.threshold == [0.25, 0.75]

    def test_unknown_metric_type_raises(self):
        with pytest.raises(ValidationError):
            MetricSpec(name="x", metric_type="not_a_type")

    def test_recsys_metric_type_accepted(self):
        spec = MetricSpec(name="precision_at_k", metric_type=MetricType.recsys)
        assert spec.metric_type == MetricType.recsys

    def test_recsys_metric_type_value(self):
        assert MetricType.recsys.value == "recsys"


class TestMonitoringPlan:
    def _plan(self, **kwargs):
        defaults = dict(
            name="test_plan",
            model_id="model_a",
            model_version="1.0",
            data_schema=TabularSchema(),
            metrics=[MetricSpec(name="accuracy")],
        )
        defaults.update(kwargs)
        return MonitoringPlan(**defaults)

    def test_basic(self):
        plan = self._plan()
        assert plan.name == "test_plan"
        assert plan.model_id == "model_a"
        assert len(plan.metrics) == 1
        assert plan.description == ""

    def test_multiple_metrics(self):
        specs = [
            MetricSpec(name="accuracy"),
            MetricSpec(name="f1"),
            MetricSpec(name="psi", feature_name="age"),
        ]
        plan = self._plan(metrics=specs)
        assert len(plan.metrics) == 3

    def test_empty_metrics_allowed(self):
        plan = self._plan(metrics=[])
        assert plan.metrics == []

    def test_serialization_roundtrip(self):
        plan = self._plan()
        d = plan.model_dump()
        plan2 = MonitoringPlan.model_validate(d)
        assert plan2.name == plan.name
        assert plan2.model_id == plan.model_id
        assert isinstance(plan2.data_schema, TabularSchema)

    def test_enable_profiling_default_false(self):
        assert self._plan().enable_profiling is False

    def test_enable_profiling_accepted(self):
        plan = self._plan(enable_profiling=True)
        assert plan.enable_profiling is True

    def test_window_default_none(self):
        assert self._plan().window is None

    def test_window_last_n(self):
        plan = self._plan(window=LastNRowsWindowConfig(n=500))
        assert plan.window.type == "last_n"
        assert plan.window.n == 500

    def test_window_time_window(self):
        cfg = TimeWindowConfig(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        plan = self._plan(window=cfg)
        assert plan.window.type == "time_window"

    def test_sampling_default_none(self):
        assert self._plan().sampling is None

    def test_sampling_random_n(self):
        plan = self._plan(sampling=RandomSamplingConfig(n=500))
        assert plan.sampling.type == "random"
        assert plan.sampling.n == 500

    def test_sampling_random_frac(self):
        plan = self._plan(sampling=RandomSamplingConfig(frac=0.1))
        assert plan.sampling.type == "random"
        assert plan.sampling.frac == 0.1

    def test_plan_with_window_and_sampling_serializes(self):
        plan = self._plan(
            window=LastNRowsWindowConfig(n=1000),
            sampling=RandomSamplingConfig(n=500),
        )
        d = plan.model_dump()
        plan2 = MonitoringPlan.model_validate(d)
        assert plan2.window.type == "last_n"
        assert plan2.sampling.type == "random"
