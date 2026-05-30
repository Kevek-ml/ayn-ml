"""Tests for ayn_ml.runner.Runner."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pd = pytest.importorskip("pandas")

from ayn_ml.core.result import MonitoringReport
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MonitoringPlan
from ayn_ml.data.source import DataSource
from ayn_ml.exceptions import SchemaError
from ayn_ml.runner import Runner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan(**kwargs) -> MonitoringPlan:
    defaults = dict(
        name="test_plan",
        model_id="model_a",
        model_version="1.0",
        data_schema=TabularSchema(),
        metrics=[MetricSpec(name="accuracy")],
    )
    defaults.update(kwargs)
    return MonitoringPlan(**defaults)


def _df(n: int = 50) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "y_pred": rng.integers(0, 2, n),
            "y_pred_proba": rng.uniform(0, 1, n),
            "age": rng.normal(40, 10, n),
        }
    )


class _ConstantDataSource(DataSource):
    """DataSource stub that returns a fixed DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def load(self, plan: MonitoringPlan) -> pd.DataFrame:
        return self._df


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestRunnerHappyPath:
    def test_no_reference_returns_report(self):
        runner = Runner()
        report = runner.run(_plan(), _df())
        assert isinstance(report, MonitoringReport)
        assert len(report.results) == 1
        assert len(report.errors) == 0

    def test_fixed_reference_passed_through(self):
        runner = Runner()
        current = _df(60)
        reference = _df(40)
        plan = _plan(metrics=[MetricSpec(name="psi", feature_name="age")])
        report = runner.run(plan, current, reference=reference)
        assert len(report.results) == 1
        assert len(report.errors) == 0

    def test_plan_fields_in_context(self):
        runner = Runner()
        plan = _plan(model_id="my_model", model_version="2.0")
        report = runner.run(plan, _df())
        assert report.context.model_id == "my_model"
        assert report.context.model_version == "2.0"

    def test_eval_timestamp_is_utc(self):
        runner = Runner()
        report = runner.run(_plan(), _df())
        ts = report.context.eval_timestamp
        assert ts.tzinfo is not None
        assert ts.tzinfo == timezone.utc

    def test_empty_metrics_list_returns_empty_results(self):
        runner = Runner()
        plan = _plan(metrics=[])
        report = runner.run(plan, _df())
        assert report.results == []
        assert report.errors == []

    def test_multiple_metrics_all_computed(self):
        runner = Runner()
        plan = _plan(metrics=[MetricSpec(name="accuracy"), MetricSpec(name="f1")])
        report = runner.run(plan, _df())
        assert len(report.results) == 2

    def test_datasource_accepted_as_current(self):
        runner = Runner()
        source = _ConstantDataSource(_df())
        report = runner.run(_plan(), source)
        assert len(report.results) == 1

    def test_datasource_accepted_as_reference(self):
        runner = Runner()
        plan = _plan(metrics=[MetricSpec(name="psi", feature_name="age")])
        report = runner.run(plan, _df(), reference=_ConstantDataSource(_df()))
        assert len(report.results) == 1

    def test_invalid_current_type_raises_schema_error(self):
        from ayn_ml.exceptions import SchemaError

        with pytest.raises(SchemaError, match="DataSource"):
            Runner().run(_plan(), [1, 2, 3])

    def test_reuse_across_runs(self):
        runner = Runner()
        plan = _plan()
        r1 = runner.run(plan, _df())
        r2 = runner.run(plan, _df())
        assert len(r1.results) == len(r2.results) == 1


# ---------------------------------------------------------------------------
# n_jobs warning
# ---------------------------------------------------------------------------


class TestNJobs:
    def test_n_jobs_1_sequential(self):
        report = Runner(n_jobs=1).run(_plan(), _df())
        assert len(report.results) == 1

    def test_n_jobs_2_parallel_same_results(self):
        plan = _plan(metrics=[MetricSpec(name="accuracy"), MetricSpec(name="f1")])
        df = _df()
        r_seq = Runner(n_jobs=1).run(plan, df)
        r_par = Runner(n_jobs=2).run(plan, df)
        assert len(r_par.results) == len(r_seq.results)
        assert len(r_par.errors) == len(r_seq.errors)

    def test_n_jobs_minus_1_all_cpus(self):
        report = Runner(n_jobs=-1).run(_plan(), _df())
        assert len(report.results) == 1

    def test_n_jobs_zero_raises(self):
        with pytest.raises(ValueError, match="n_jobs"):
            Runner(n_jobs=0)

    def test_n_jobs_minus_2_raises(self):
        with pytest.raises(ValueError, match="n_jobs"):
            Runner(n_jobs=-2)


# ---------------------------------------------------------------------------
# _apply_window branches
# ---------------------------------------------------------------------------


class TestApplyWindow:
    def test_last_n_window(self):
        from ayn_ml.core.data_selection import LastNRowsWindowConfig

        runner = Runner()
        plan = _plan(
            metrics=[],
            window=LastNRowsWindowConfig(n=10),
        )
        df = _df(50)
        report = runner.run(plan, df)
        # No error means the window was applied without crash; context is built
        assert report.context is not None

    def test_plan_window_is_sole_configuration(self):
        from ayn_ml.core.data_selection import LastNRowsWindowConfig

        plan = _plan(metrics=[], window=LastNRowsWindowConfig(n=10))
        report = Runner().run(plan, _df(50))
        assert report.context is not None

    def test_time_window_without_timestamp_col_raises_schema_error(self):
        from datetime import timezone

        from ayn_ml.core.data_selection import TimeWindowConfig

        plan = _plan(
            metrics=[],
            window=TimeWindowConfig(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            ),
        )  # TabularSchema() default — timestamp_col=None
        with pytest.raises(SchemaError, match="timestamp_col"):
            Runner().run(plan, _df())


# ---------------------------------------------------------------------------
# _filter_model
# ---------------------------------------------------------------------------


class TestFilterModel:
    def test_model_id_col_filters_rows(self):
        schema = TabularSchema(model_id_col="model_id")
        df = pd.DataFrame(
            {
                "y_true": [1, 0, 1, 0],
                "y_pred": [1, 0, 0, 1],
                "model_id": ["model_a", "model_b", "model_a", "model_b"],
            }
        )
        plan = _plan(data_schema=schema, model_id="model_a", metrics=[MetricSpec(name="accuracy")])
        report = Runner().run(plan, df)
        # accuracy computed on 2 rows from model_a — no crash, result present
        assert len(report.results) == 1

    def test_missing_model_id_col_raises_in_strict_mode(self):
        schema = TabularSchema(model_id_col="model_id")
        df = _df()  # no model_id column
        plan = _plan(data_schema=schema)
        with pytest.raises(SchemaError, match="model_id_col"):
            Runner().run(plan, df)

    def test_missing_model_id_col_warns_in_lenient_mode(self):
        schema = TabularSchema(model_id_col="model_id")
        df = _df()  # no model_id column
        plan = _plan(data_schema=schema)
        report = Runner(strict=False).run(plan, df)
        assert len(report.results) == 1

    def test_model_version_col_filters_rows(self):
        schema = TabularSchema(model_version_col="version")
        df = pd.DataFrame(
            {
                "y_true": [1, 0, 1],
                "y_pred": [1, 0, 1],
                "version": ["1.0", "2.0", "1.0"],
            }
        )
        plan = _plan(data_schema=schema, model_version="1.0", metrics=[MetricSpec(name="accuracy")])
        report = Runner().run(plan, df)
        assert len(report.results) == 1


# ---------------------------------------------------------------------------
# _build_context — timestamp handling
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_period_bounds_set_when_timestamp_col_present(self):
        schema = TabularSchema(timestamp_col="ts")
        df = _df(10)
        df["ts"] = pd.date_range("2026-01-01", periods=10, freq="h")
        plan = _plan(data_schema=schema, metrics=[])
        report = Runner().run(plan, df)
        assert isinstance(report.context.period_start, datetime)
        assert isinstance(report.context.period_end, datetime)
        assert report.context.period_start <= report.context.period_end

    def test_period_bounds_none_when_no_timestamp_col(self):
        plan = _plan(data_schema=TabularSchema(), metrics=[])
        report = Runner().run(plan, _df())
        assert report.context.period_start is None
        assert report.context.period_end is None

    def test_period_bounds_are_stdlib_datetime_not_pandas_timestamp(self):
        schema = TabularSchema(timestamp_col="ts")
        df = _df(5)
        df["ts"] = pd.date_range("2026-03-01", periods=5, freq="D")
        plan = _plan(data_schema=schema, metrics=[])
        report = Runner().run(plan, df)
        # Must be stdlib datetime, not pandas.Timestamp (Pydantic requirement)
        assert type(report.context.period_start) is datetime
        assert type(report.context.period_end) is datetime


# ---------------------------------------------------------------------------
# Metric loop — error isolation
# ---------------------------------------------------------------------------


class TestMetricLoopIsolation:
    def test_ayn_error_is_caught_stored_not_raised(self):
        from unittest.mock import MagicMock, patch

        fake_metric = MagicMock()
        fake_metric.requires_reference = False
        fake_metric.compute.side_effect = SchemaError("boom")

        with patch("ayn_ml.runner.get_metric", return_value=fake_metric):
            plan = _plan(metrics=[MetricSpec(name="accuracy")])
            report = Runner().run(plan, _df())

        assert len(report.errors) == 1
        assert report.errors[0].metric_name == "accuracy"
        assert report.errors[0].error_type == "SchemaError"

    def test_unexpected_exception_is_caught_stored_not_raised(self):
        from unittest.mock import MagicMock, patch

        fake_metric = MagicMock()
        fake_metric.requires_reference = False
        fake_metric.compute.side_effect = ValueError("unexpected boom")

        with patch("ayn_ml.runner.get_metric", return_value=fake_metric):
            plan = _plan(metrics=[MetricSpec(name="accuracy")])
            report = Runner().run(plan, _df())

        assert len(report.errors) == 1
        assert report.errors[0].error_type == "ValueError"

    def test_failing_metric_does_not_abort_subsequent_metrics(self):
        from unittest.mock import MagicMock, patch

        from ayn_ml.core.result import MetricResult

        fail_metric = MagicMock()
        fail_metric.requires_reference = False
        fail_metric.compute.side_effect = SchemaError("first fails")

        ok_result = MagicMock(spec=MetricResult)
        ok_metric = MagicMock()
        ok_metric.requires_reference = False
        ok_metric.compute.return_value = ok_result

        side_effects = [fail_metric, ok_metric]

        with patch("ayn_ml.runner.get_metric", side_effect=side_effects):
            plan = _plan(
                metrics=[
                    MetricSpec(name="bad_metric"),
                    MetricSpec(name="accuracy"),
                ]
            )
            report = Runner().run(plan, _df())

        assert len(report.errors) == 1
        assert len(report.results) == 1

    def test_requires_reference_without_reference_skipped_as_error(self):
        plan = _plan(metrics=[MetricSpec(name="psi", feature_name="age")])
        report = Runner().run(plan, _df(), reference=None)
        assert len(report.errors) == 1
        assert report.errors[0].error_type == "MissingReferenceError"
        assert report.errors[0].metric_name == "psi"


# ---------------------------------------------------------------------------
# store.write() failure handling
# ---------------------------------------------------------------------------


class TestStoreWriteFailure:
    def test_store_write_called_with_report(self):
        store = MagicMock()
        plan = _plan()
        report = Runner().run(plan, _df(), store=store)
        store.write.assert_called_once_with(report)

    def test_store_write_exception_does_not_abort_run(self, caplog):
        store = MagicMock()
        store.write.side_effect = RuntimeError("store down")
        with caplog.at_level(logging.WARNING, logger="ayn_ml.runner"):
            report = Runner().run(_plan(), _df(), store=store)
        assert isinstance(report, MonitoringReport)
        assert any("store.write" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# model argument warning
# ---------------------------------------------------------------------------


class TestModelArgWarning:
    def test_model_arg_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ayn_ml.runner"):
            Runner().run(_plan(), _df(), model=object())
        assert any("model" in m and "Phase 5" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# alert_rules warning
# ---------------------------------------------------------------------------


class TestAlertRulesWarning:
    def test_alert_rule_unknown_metric_logs_warning(self, caplog):
        """Alert rule referencing a metric not in the plan → warning, no crash."""
        from ayn_ml.core.alert import AlertRule, ThresholdPolicy

        rule = AlertRule(metric_name="nonexistent_metric", policy=ThresholdPolicy(), channels=[])
        with caplog.at_level(logging.WARNING, logger="ayn_ml.runner"):
            Runner().run(_plan(), _df(), alert_rules=[rule])
        assert any("unknown metric" in m.lower() for m in caplog.messages)


# ---------------------------------------------------------------------------
# sinks dispatch
# ---------------------------------------------------------------------------


class TestSinksDispatch:
    def test_sink_write_called_on_every_run(self):
        """Unconditional sinks receive write() on every run."""
        sink = MagicMock()
        report = Runner().run(_plan(), _df(), sinks=[sink])
        sink.write.assert_called_once_with(report)

    def test_sink_write_error_logs_warning(self, caplog):
        """A failing sink write is caught and logged as a warning."""
        bad_sink = MagicMock()
        bad_sink.write.side_effect = RuntimeError("network error")
        with caplog.at_level(logging.WARNING, logger="ayn_ml.runner"):
            Runner().run(_plan(), _df(), sinks=[bad_sink])
        assert any("sink.write() failed" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# empty DataFrame after model filtering
# ---------------------------------------------------------------------------


class TestEmptyDataFrameAfterFiltering:
    def test_empty_current_after_model_filter_logs_warning(self, caplog):
        schema = TabularSchema(model_id_col="model_id")
        df = pd.DataFrame(
            {
                "y_true": [1, 0],
                "y_pred": [1, 0],
                "model_id": ["other_model", "other_model"],
            }
        )
        plan = _plan(data_schema=schema, model_id="model_a", metrics=[])
        with caplog.at_level(logging.WARNING, logger="ayn_ml.runner"):
            report = Runner().run(plan, df)
        assert isinstance(report, MonitoringReport)
        assert any("empty" in m.lower() for m in caplog.messages)
