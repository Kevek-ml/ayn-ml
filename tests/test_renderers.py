"""Tests for ayn_ml.renderers — HtmlRenderer, PlotlyBackend, NoChartBackend."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ayn_ml.core.result import ExecutionContext, FiredAlert, MetricError, MetricResult, MonitoringReport
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType, MonitoringPlan
from ayn_ml.renderers.html import HtmlRenderer, _fmt_value, _metric_rows, _profile_data, _row_class, _status_icon
from ayn_ml.renderers.no_chart import NoChartBackend
from ayn_ml.renderers.plotly import PlotlyBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _plan(name: str = "test_plan", model_id: str = "m", model_version: str = "1") -> MonitoringPlan:
    return MonitoringPlan(
        name=name,
        model_id=model_id,
        model_version=model_version,
        data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
        metrics=[MetricSpec(name="accuracy")],
    )


def _ctx(
    model_id: str = "m",
    model_version: str = "1",
    ts: datetime | None = None,
    n_current: int = 100,
) -> ExecutionContext:
    return ExecutionContext(
        model_id=model_id,
        model_version=model_version,
        eval_timestamp=ts or datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        n_current=n_current,
    )


def _result(
    name: str = "accuracy",
    value: float = 0.92,
    status: bool | None = True,
    threshold: float | None = 0.8,
    metric_type: MetricType | None = MetricType.performance,
) -> MetricResult:
    return MetricResult(
        spec=MetricSpec(name=name, threshold=threshold, metric_type=metric_type),
        value=value,
        status=status,
    )


def _report(
    results: list[MetricResult] | None = None,
    errors: list[MetricError] | None = None,
    fired_alerts: list[FiredAlert] | None = None,
    profile: dict | None = None,
    ts: datetime | None = None,
) -> MonitoringReport:
    return MonitoringReport(
        plan=_plan(),
        context=_ctx(ts=ts),
        results=results or [_result()],
        errors=errors or [],
        fired_alerts=fired_alerts or [],
        profile=profile,
    )


# ---------------------------------------------------------------------------
# NoChartBackend
# ---------------------------------------------------------------------------


class TestNoChartBackend:
    def test_include_scripts_returns_empty_string(self):
        assert NoChartBackend().include_scripts() == ""

    def test_bar_chart_returns_empty_string(self):
        result = NoChartBackend().bar_chart(["a", "b"], [1.0, 2.0], "Title")
        assert result == ""

    def test_timeseries_returns_empty_string(self):
        result = NoChartBackend().timeseries(["2024-01", "2024-02"], [0.9, 0.85], "Accuracy")
        assert result == ""

    def test_timeseries_with_threshold_returns_empty_string(self):
        result = NoChartBackend().timeseries(["2024-01"], [0.9], "Accuracy", threshold=0.8)
        assert result == ""


# ---------------------------------------------------------------------------
# PlotlyBackend
# ---------------------------------------------------------------------------


class TestPlotlyBackend:
    def test_include_scripts_contains_plotly(self):
        scripts = PlotlyBackend().include_scripts()
        assert "plotly" in scripts.lower()
        assert "<script" in scripts

    def test_bar_chart_returns_html_string(self):
        html = PlotlyBackend().bar_chart(["accuracy", "psi"], [0.92, 0.08], "Metrics")
        assert isinstance(html, str)
        assert len(html) > 0

    def test_bar_chart_contains_div(self):
        html = PlotlyBackend().bar_chart(["a"], [1.0], "Test")
        assert "<div" in html

    def test_bar_chart_empty_inputs_returns_empty(self):
        assert PlotlyBackend().bar_chart([], [], "Empty") == ""

    def test_timeseries_returns_html_string(self):
        html = PlotlyBackend().timeseries(["2024-01-01", "2024-01-02"], [0.9, 0.88], "Accuracy")
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<div" in html

    def test_timeseries_empty_inputs_returns_empty(self):
        assert PlotlyBackend().timeseries([], [], "Empty") == ""

    def test_timeseries_with_threshold_returns_html(self):
        html = PlotlyBackend().timeseries(
            ["2024-01-01", "2024-01-02"],
            [0.9, 0.75],
            "Accuracy",
            threshold=0.8,
        )
        assert "<div" in html


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestFmtValue:
    def test_none_returns_dash(self):
        assert _fmt_value(None) == "—"

    def test_float_uses_4sig(self):
        assert _fmt_value(0.123456) == "0.1235"

    def test_int_returns_string(self):
        assert _fmt_value(42) == "42"

    def test_string_returned_as_is(self):
        assert _fmt_value("high") == "high"


class TestStatusIcon:
    def test_true_returns_checkmark(self):
        assert _status_icon(True) == "✓"

    def test_false_returns_cross(self):
        assert _status_icon(False) == "✗"

    def test_none_returns_dash(self):
        assert _status_icon(None) == "—"


class TestRowClass:
    def test_pass_returns_row_pass(self):
        assert _row_class(True) == "row-pass"

    def test_fail_returns_row_fail(self):
        assert _row_class(False) == "row-fail"

    def test_none_returns_empty(self):
        assert _row_class(None) == ""


class TestMetricRows:
    def test_returns_one_row_per_result(self):
        results = [_result("accuracy"), _result("psi", value=0.08, status=True)]
        rows = _metric_rows(results)
        assert len(rows) == 2

    def test_row_contains_expected_keys(self):
        rows = _metric_rows([_result()])
        row = rows[0]
        assert "name" in row
        assert "value" in row
        assert "status_icon" in row
        assert "row_class" in row
        assert "threshold" in row
        assert "effect_size" in row

    def test_pass_status_icon(self):
        rows = _metric_rows([_result(status=True)])
        assert rows[0]["status_icon"] == "✓"

    def test_fail_status_icon(self):
        rows = _metric_rows([_result(status=False)])
        assert rows[0]["status_icon"] == "✗"


class TestProfileData:
    def test_empty_profile_returns_empty(self):
        stat_names, rows = _profile_data({})
        assert stat_names == []
        assert rows == []

    def test_stat_names_collected(self):
        profile = {"col_a": {"mean": 1.0, "std": 0.1}, "col_b": {"mean": 2.0, "std": 0.2}}
        stat_names, _ = _profile_data(profile)
        assert "mean" in stat_names
        assert "std" in stat_names

    def test_rows_sorted_by_column_name(self):
        profile = {"z_col": {"mean": 1.0}, "a_col": {"mean": 2.0}}
        _, rows = _profile_data(profile)
        assert rows[0]["col"] == "a_col"
        assert rows[1]["col"] == "z_col"


# ---------------------------------------------------------------------------
# HtmlRenderer.render() — single report
# ---------------------------------------------------------------------------


class TestHtmlRendererRender:
    def test_returns_string(self):
        html = HtmlRenderer(charts=NoChartBackend()).render(_report())
        assert isinstance(html, str)

    def test_contains_doctype(self):
        html = HtmlRenderer(charts=NoChartBackend()).render(_report())
        assert "<!DOCTYPE html>" in html

    def test_contains_plan_name(self):
        report = _report()
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "test_plan" in html

    def test_contains_metric_names(self):
        report = _report(results=[_result("accuracy"), _result("psi", value=0.08)])
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "accuracy" in html
        assert "psi" in html

    def test_contains_metric_values(self):
        report = _report(results=[_result("accuracy", value=0.923456)])
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "0.923" in html  # 4-sig format

    def test_pass_indicator_present(self):
        report = _report(results=[_result(status=True)])
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "✓" in html

    def test_fail_indicator_present(self):
        report = _report(results=[_result(status=False)])
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "✗" in html

    def test_fired_alerts_section_present(self):
        alerts = [FiredAlert(metric_name="psi", policy_type="threshold", details={"value": 0.15})]
        report = _report(fired_alerts=alerts)
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "Fired Alerts" in html
        assert "psi" in html

    def test_no_alert_section_when_no_alerts(self):
        report = _report(fired_alerts=[])
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "Fired Alerts" not in html

    def test_errors_section_present(self):
        errors = [MetricError(metric_name="auc", error_type="SchemaError", message="col missing")]
        report = _report(errors=errors)
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "auc" in html
        assert "col missing" in html

    def test_no_errors_section_when_no_errors(self):
        report = _report(errors=[])
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "Errors" not in html

    def test_profile_section_present_when_profiling_enabled(self):
        profile = {"income": {"mean": 50000.0, "std": 15000.0, "null_rate": 0.0}}
        report = _report(profile=profile)
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "Column Profile" in html
        assert "income" in html

    def test_no_profile_section_when_no_profiling(self):
        report = _report(profile=None)
        html = HtmlRenderer(charts=NoChartBackend()).render(report)
        assert "Column Profile" not in html

    def test_default_backend_is_plotly(self):
        renderer = HtmlRenderer()
        assert isinstance(renderer._charts, PlotlyBackend)

    def test_no_chart_backend_produces_no_chart_html(self):
        renderer = HtmlRenderer(charts=NoChartBackend())
        report = _report(results=[_result("accuracy")])
        html = renderer.render(report)
        # No chart section since backend returns empty string
        assert "Metric Values" not in html

    def test_plotly_backend_injects_script_tag(self):
        renderer = HtmlRenderer(charts=PlotlyBackend())
        report = _report(results=[_result("accuracy")])
        html = renderer.render(report)
        assert "plotly" in html.lower()
        assert "<script" in html


# ---------------------------------------------------------------------------
# HtmlRenderer.render_history() — multi-report time series
# ---------------------------------------------------------------------------


class TestHtmlRendererRenderHistory:
    def _reports_sequence(self, n: int = 3) -> list[MonitoringReport]:
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        reports = []
        for i in range(n):
            ts = base_ts + timedelta(days=i)
            value = 0.95 - i * 0.02
            status = value >= 0.9
            reports.append(_report(results=[_result("accuracy", value=value, status=status)], ts=ts))
        return reports

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="at least one"):
            HtmlRenderer(charts=NoChartBackend()).render_history([])

    def test_returns_html_string(self):
        reports = self._reports_sequence()
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_plan_name(self):
        reports = self._reports_sequence()
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        assert "test_plan" in html

    def test_contains_metric_name(self):
        reports = self._reports_sequence()
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        assert "accuracy" in html

    def test_contains_run_count(self):
        reports = self._reports_sequence(n=4)
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        assert "4" in html  # n_runs displayed

    def test_latest_results_section_present(self):
        reports = self._reports_sequence()
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        assert "Latest Results" in html

    def test_accepts_single_report(self):
        reports = self._reports_sequence(n=1)
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        assert isinstance(html, str)

    def test_sorts_by_timestamp_internally(self):
        """Reports in reverse order are sorted correctly — last_ts is the most recent."""
        reports = list(reversed(self._reports_sequence(n=3)))
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        # Just confirm it renders without error and contains the date range
        assert "2024-01-01" in html
        assert "2024-01-03" in html

    def test_plotly_backend_generates_timeseries_charts(self):
        reports = self._reports_sequence(n=3)
        html = HtmlRenderer(charts=PlotlyBackend()).render_history(reports)
        assert "Metric Trends" in html
        assert "<div" in html

    def test_no_chart_backend_skips_trend_section(self):
        reports = self._reports_sequence(n=3)
        html = HtmlRenderer(charts=NoChartBackend()).render_history(reports)
        # No timeseries charts → section skipped
        assert "Metric Trends" not in html
