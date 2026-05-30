from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ayn_ml.core.result import (
    ExecutionContext,
    FiredAlert,
    MetricError,
    MetricResult,
    MonitoringReport,
)
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType, MonitoringPlan


def make_context(**kwargs) -> ExecutionContext:
    defaults = dict(
        model_id="model_a",
        model_version="1.0",
        eval_timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return ExecutionContext(**defaults)


def make_spec(**kwargs) -> MetricSpec:
    defaults = dict(name="accuracy", metric_type=MetricType.performance)
    defaults.update(kwargs)
    return MetricSpec(**defaults)


def make_plan() -> MonitoringPlan:
    return MonitoringPlan(
        name="test_plan",
        model_id="model_a",
        model_version="1.0",
        data_schema=TabularSchema(),
        metrics=[make_spec()],
    )


class TestExecutionContext:
    def test_basic(self):
        ctx = make_context()
        assert ctx.model_id == "model_a"
        assert ctx.period_start is None

    def test_with_period(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 15, tzinfo=timezone.utc)
        ctx = make_context(period_start=start, period_end=end)
        assert ctx.period_start == start
        assert ctx.period_end == end

    def test_frozen(self):
        ctx = make_context()
        with pytest.raises(ValidationError):
            ctx.model_id = "other"

    def test_requires_model_id(self):
        with pytest.raises(ValidationError):
            ExecutionContext(
                model_version="1.0",
                eval_timestamp=datetime.now(timezone.utc),
            )


class TestMetricResult:
    def test_pass(self):
        result = MetricResult(spec=make_spec(threshold=0.8), value=0.9, status=True)
        assert result.value == 0.9
        assert result.status is True

    def test_fail(self):
        result = MetricResult(spec=make_spec(threshold=0.8), value=0.7, status=False)
        assert result.status is False

    def test_no_threshold(self):
        result = MetricResult(spec=make_spec(), value=0.85)
        assert result.status is None

    def test_with_conf_interval(self):
        result = MetricResult(spec=make_spec(), value=0.85, conf_interval=(0.80, 0.90))
        assert result.conf_interval == (0.80, 0.90)

    def test_effect_size_defaults_to_none(self):
        result = MetricResult(spec=make_spec(), value=0.85)
        assert result.effect_size is None
        assert result.effect_size_label is None

    def test_effect_size_round_trips_via_model_dump(self):
        result = MetricResult(spec=make_spec(), value=0.5, effect_size=0.42, effect_size_label="cohen_d")
        d = result.model_dump()
        assert d["effect_size"] == 0.42
        assert d["effect_size_label"] == "cohen_d"

    def test_string_value(self):
        result = MetricResult(spec=make_spec(metric_type=MetricType.statistics), value="high")
        assert result.value == "high"

    def test_no_context_field(self):
        result = MetricResult(spec=make_spec(), value=0.85)
        assert not hasattr(result, "context")


class TestMonitoringReport:
    def _make_report(self, **kwargs):
        plan = make_plan()
        ctx = make_context()
        result = MetricResult(spec=make_spec(), value=0.85)
        defaults = dict(plan=plan, context=ctx, results=[result], errors=[])
        defaults.update(kwargs)
        return MonitoringReport(**defaults)

    def test_basic(self):
        report = self._make_report()
        assert len(report.results) == 1
        assert report.errors == []
        assert report.fired_alerts == []

    def test_context_lives_on_report_not_result(self):
        report = self._make_report()
        assert report.context.model_id == "model_a"
        assert not hasattr(report.results[0], "context")

    def test_with_errors(self):
        error = MetricError("psi", "InsufficientDataError", "only 3 rows")
        report = self._make_report(errors=[error])
        assert len(report.errors) == 1
        assert report.errors[0].metric_name == "psi"

    def test_with_fired_alerts(self):
        alert = FiredAlert("accuracy", "threshold", {"threshold": 0.8, "value": 0.7})
        report = self._make_report(fired_alerts=[alert])
        assert len(report.fired_alerts) == 1
        assert report.fired_alerts[0].policy_type == "threshold"

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        assert "plan" in d
        assert "context" in d
        assert "results" in d
        assert "errors" in d
        assert "fired_alerts" in d
        assert d["results"][0]["value"] == 0.85

    def test_to_dict_with_error(self):
        error = MetricError("f1", "ValueError", "oops")
        report = self._make_report(errors=[error])
        d = report.to_dict()
        assert d["errors"][0]["metric_name"] == "f1"
        assert d["errors"][0]["message"] == "oops"

    def test_to_dict_with_fired_alert(self):
        alert = FiredAlert("accuracy", "change", {"pct": 0.12})
        report = self._make_report(fired_alerts=[alert])
        d = report.to_dict()
        assert d["fired_alerts"][0]["policy_type"] == "change"
        assert "feature_name" in d["fired_alerts"][0]

    def test_to_dict_fired_alert_feature_name_populated(self):
        alert = FiredAlert("psi", "threshold", {"threshold": 0.2, "value": 0.5}, feature_name="age")
        report = self._make_report(fired_alerts=[alert])
        d = report.to_dict()
        assert d["fired_alerts"][0]["feature_name"] == "age"

    def test_to_dataframe_uses_report_context(self):
        pd = pytest.importorskip("pandas")
        report = self._make_report()
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["metric_name"] == "accuracy"
        assert df.iloc[0]["value"] == 0.85
        assert df.iloc[0]["model_id"] == "model_a"

    def test_to_dataframe_with_none_metric_type(self):
        pd = pytest.importorskip("pandas")
        spec = MetricSpec(name="accuracy")  # metric_type=None
        result = MetricResult(spec=spec, value=0.9, status=True)
        report = self._make_report(results=[result])
        df = report.to_dataframe()
        assert pd.isna(df.iloc[0]["metric_type"])

    def test_to_dataframe_includes_effect_size(self):
        pytest.importorskip("pandas")
        result = MetricResult(spec=make_spec(), value=0.5, effect_size=0.75, effect_size_label="cohen_d")
        report = self._make_report(results=[result])
        df = report.to_dataframe()
        assert "effect_size" in df.columns
        assert df.iloc[0]["effect_size"] == 0.75
        assert df.iloc[0]["effect_size_label"] == "cohen_d"

    def test_to_dataframe_effect_size_none_when_absent(self):
        pd = pytest.importorskip("pandas")
        report = self._make_report()
        df = report.to_dataframe()
        assert pd.isna(df.iloc[0]["effect_size"])
        assert pd.isna(df.iloc[0]["effect_size_label"])
