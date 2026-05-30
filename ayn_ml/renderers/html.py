"""HTML renderer for MonitoringReport objects.

``HtmlRenderer`` produces self-contained HTML documents from one or more
``MonitoringReport`` objects using Jinja2 templates.  Jinja2 is a core
dependency of ayn-ml, so no import guard is needed.

Usage::

    from ayn_ml.renderers import HtmlRenderer, NoChartBackend

    renderer = HtmlRenderer()                          # Plotly charts (default)
    html = renderer.render(report)                     # single-run snapshot
    html = renderer.render_history([r1, r2, r3])       # time-series dashboard

    # Headless / offline — tables only, no charts
    renderer = HtmlRenderer(charts=NoChartBackend())
    html = renderer.render(report)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jinja2 import Template

from ayn_ml.renderers.plotly import PlotlyBackend

if TYPE_CHECKING:
    from ayn_ml.core.result import MetricResult, MonitoringReport
    from ayn_ml.renderers.base import ChartBackend

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 templates (inline — no file I/O, no package-data configuration)
# ---------------------------------------------------------------------------

_BASE_STYLE = """
<style>
  body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1280px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #212529;
    background: #f8f9fa;
  }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #dee2e6; padding-bottom: 0.25rem; }
  .meta { color: #6c757d; font-size: 0.875rem; margin-bottom: 1.5rem; }
  .meta span { margin-right: 1.5rem; }
  table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; background: #fff;
          border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.07); }
  th { background: #e9ecef; padding: 0.6rem 1rem; text-align: left; font-size: 0.85rem;
       font-weight: 600; letter-spacing: 0.02em; color: #495057; }
  td { padding: 0.5rem 1rem; font-size: 0.875rem; border-top: 1px solid #dee2e6; }
  .pass td.status-cell { color: #155724; font-weight: 700; }
  .fail td.status-cell { color: #721c24; font-weight: 700; }
  .row-pass { background: #f0fff4; }
  .row-fail { background: #fff5f5; }
  .alert-box {
    padding: 0.75rem 1rem; background: #fff3cd; border: 1px solid #ffc107;
    border-left: 4px solid #ffc107; border-radius: 4px; margin: 0.5rem 0;
    font-size: 0.875rem;
  }
  .alert-box.error { background: #f8d7da; border-color: #dc3545; border-left-color: #dc3545; }
  .chart-section { background: #fff; border-radius: 6px; padding: 0.5rem;
                   box-shadow: 0 1px 3px rgba(0,0,0,.07); margin-bottom: 1.5rem; }
  code { background: #e9ecef; padding: 0.1em 0.4em; border-radius: 3px;
         font-size: 0.85em; }
</style>
"""

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ plan_name }} — Monitoring Report</title>
  {{ scripts }}
  {{ style }}
</head>
<body>
  <h1>{{ plan_name }}</h1>
  <div class="meta">
    <span><strong>Model:</strong> {{ model_id }} v{{ model_version }}</span>
    <span><strong>Run ID:</strong> <code>{{ run_id }}</code></span>
    <span><strong>Evaluated:</strong> {{ eval_timestamp }}</span>
    {% if period_start %}<span><strong>Period:</strong> {{ period_start }} → {{ period_end }}</span>{% endif %}
    <span><strong>Rows:</strong> {{ n_current }}{% if n_reference %} / {{ n_reference }} ref{% endif %}</span>
  </div>

  {% if fired_alerts %}
  <h2>🔴 Fired Alerts ({{ fired_alerts|length }})</h2>
  {% for alert in fired_alerts %}
  <div class="alert-box">
    <strong>{{ alert.metric_name }}</strong> — policy: {{ alert.policy_type }}
    {% if alert.details %}
    &nbsp;&nbsp;value: <strong>{{ alert.details.value }}</strong>
    &nbsp;&nbsp;threshold: {{ alert.details.threshold }}
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  <h2>Metric Results</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th><th>Type</th><th>Feature</th><th>Value</th>
        <th>Status</th><th>Threshold</th><th>Effect Size</th>
      </tr>
    </thead>
    <tbody>
      {% for row in metric_rows %}
      <tr class="{{ row.row_class }}">
        <td>{{ row.name }}</td>
        <td>{{ row.metric_type }}</td>
        <td>{{ row.feature_name }}</td>
        <td>{{ row.value }}</td>
        <td class="status-cell">{{ row.status_icon }}</td>
        <td>{{ row.threshold }}</td>
        <td>{{ row.effect_size }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  {% if error_rows %}
  <h2>Errors ({{ error_rows|length }})</h2>
  {% for e in error_rows %}
  <div class="alert-box error">
    <strong>{{ e.metric_name }}</strong> [{{ e.error_type }}]: {{ e.message }}
  </div>
  {% endfor %}
  {% endif %}

  {% if profile_rows %}
  <h2>Column Profile</h2>
  <table>
    <thead>
      <tr>
        <th>Column</th>
        {% for stat in profile_stat_names %}
        <th>{{ stat }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in profile_rows %}
      <tr>
        <td><code>{{ row.col }}</code></td>
        {% for stat in profile_stat_names %}
        <td>{{ row.stats.get(stat, "—") }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  {% if chart_html %}
  <h2>Metric Values</h2>
  <div class="chart-section">{{ chart_html }}</div>
  {% endif %}
</body>
</html>
"""

_HISTORY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ plan_name }} — History Dashboard</title>
  {{ scripts }}
  {{ style }}
</head>
<body>
  <h1>{{ plan_name }}</h1>
  <div class="meta">
    <span><strong>Model:</strong> {{ model_id }} v{{ model_version }}</span>
    <span><strong>Runs:</strong> {{ n_runs }}</span>
    <span><strong>Range:</strong> {{ first_ts }} → {{ last_ts }}</span>
  </div>

  {% if timeseries_charts %}
  <h2>Metric Trends</h2>
  {% for chart in timeseries_charts %}
  <div class="chart-section">{{ chart }}</div>
  {% endfor %}
  {% endif %}

  <h2>Latest Results</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>Feature</th><th>Latest Value</th><th>Status</th><th>Threshold</th></tr>
    </thead>
    <tbody>
      {% for row in latest_rows %}
      <tr class="{{ row.row_class }}">
        <td>{{ row.name }}</td>
        <td>{{ row.feature_name }}</td>
        <td>{{ row.value }}</td>
        <td class="status-cell">{{ row.status_icon }}</td>
        <td>{{ row.threshold }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_value(value: Any) -> str:
    """Format a metric value for display.

    Args:
        value: Raw metric value (float, int, str, or None).

    Returns:
        Formatted string — 4 significant digits for floats, raw string
        for strings, ``"—"`` for None.
    """
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _fmt_effect(result: MetricResult) -> str:
    """Format effect size and label for display.

    Args:
        result: MetricResult with optional effect_size fields.

    Returns:
        Formatted effect-size string, or ``"—"`` when not available.
    """
    if result.effect_size is None:
        return "—"
    label = f" ({result.effect_size_label})" if result.effect_size_label else ""
    return f"{result.effect_size:.4g}{label}"


def _row_class(status: bool | None) -> str:
    """Return the CSS row class for a given status.

    Args:
        status: True = pass, False = fail, None = no threshold.

    Returns:
        CSS class string.
    """
    if status is True:
        return "row-pass"
    if status is False:
        return "row-fail"
    return ""


def _status_icon(status: bool | None) -> str:
    """Return a unicode status indicator.

    Args:
        status: True = pass, False = fail, None = no threshold.

    Returns:
        ``"✓"``, ``"✗"``, or ``"—"``.
    """
    if status is True:
        return "✓"
    if status is False:
        return "✗"
    return "—"


def _metric_rows(results: list[MetricResult]) -> list[dict[str, Any]]:
    """Convert MetricResult objects into template-ready dicts.

    Args:
        results: List of MetricResult objects.

    Returns:
        List of dicts with display-ready string fields.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.spec.name,
                "metric_type": r.spec.metric_type.value if r.spec.metric_type else "",
                "feature_name": r.spec.feature_name or "",
                "value": _fmt_value(r.value),
                "status_icon": _status_icon(r.status),
                "row_class": _row_class(r.status),
                "threshold": str(r.spec.threshold) if r.spec.threshold is not None else "—",
                "effect_size": _fmt_effect(r),
            }
        )
    return rows


def _profile_data(
    profile: dict[str, dict[str, float | int | str | None]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract stat names and per-column rows from a profile dict.

    Args:
        profile: ``MonitoringReport.profile`` mapping col → {stat → value}.

    Returns:
        Tuple of ``(stat_names, rows)`` where each row is a dict with
        ``col`` and ``stats`` keys.
    """
    if not profile:
        return [], []
    # Collect all stat names across columns, preserving insertion order
    stat_names: list[str] = []
    seen: set[str] = set()
    for stats in profile.values():
        for k in stats:
            if k not in seen:
                stat_names.append(k)
                seen.add(k)

    rows = []
    for col, stats in sorted(profile.items()):
        rows.append(
            {
                "col": col,
                "stats": {k: _fmt_value(v) for k, v in stats.items()},
            }
        )
    return stat_names, rows


# ---------------------------------------------------------------------------
# HtmlRenderer
# ---------------------------------------------------------------------------


class HtmlRenderer:
    """HTML report renderer for MonitoringReport objects.

    Produces complete, self-contained HTML documents.  The default chart
    backend is ``PlotlyBackend`` — swap in ``NoChartBackend`` for headless
    environments that cannot load the Plotly CDN.

    Args:
        charts: Chart backend to use for generating chart HTML fragments.
            Defaults to ``PlotlyBackend()``.

    Example::

        from ayn_ml.renderers import HtmlRenderer, NoChartBackend

        # With interactive Plotly charts (default)
        renderer = HtmlRenderer()
        html = renderer.render(report)

        # Tables only — useful for email bodies or offline environments
        renderer = HtmlRenderer(charts=NoChartBackend())
        html = renderer.render(report)
    """

    def __init__(self, charts: ChartBackend | None = None) -> None:
        """Initialise the renderer.

        Args:
            charts: Chart backend.  Defaults to ``PlotlyBackend()``.
        """
        self._charts: ChartBackend = charts if charts is not None else PlotlyBackend()

    def render(self, report: MonitoringReport) -> str:
        """Render a single monitoring report snapshot as a complete HTML document.

        The document contains:

        - Run metadata header (model, run ID, timestamp, row counts).
        - Fired alerts section (if any alerts triggered).
        - Metric results table (name, value, pass/fail status, threshold,
          effect size).
        - Errors section (if any metrics failed).
        - Column profile table (if ``enable_profiling=True`` in the plan).
        - Bar chart of metric values (when using ``PlotlyBackend``).

        Args:
            report: Completed ``MonitoringReport`` from a Runner execution.

        Returns:
            Complete HTML document as a string.
        """
        ctx = report.context
        plan = report.plan

        metric_rows = _metric_rows(report.results)
        error_rows = [
            {"metric_name": e.metric_name, "error_type": e.error_type, "message": e.message} for e in report.errors
        ]

        profile_stat_names: list[str] = []
        profile_rows: list[dict[str, Any]] = []
        if report.profile:
            profile_stat_names, profile_rows = _profile_data(report.profile)

        # Bar chart — numeric metrics only
        chart_labels = [r.spec.name for r in report.results if isinstance(r.value, int | float)]
        chart_values = [float(r.value) for r in report.results if isinstance(r.value, int | float)]  # type: ignore[arg-type]
        chart_html = self._charts.bar_chart(chart_labels, chart_values, "Metric Values") if chart_labels else ""

        fired_alerts = [
            {
                "metric_name": fa.metric_name,
                "policy_type": fa.policy_type,
                "details": fa.details,
            }
            for fa in report.fired_alerts
        ]

        tmpl = Template(_REPORT_TEMPLATE)
        return tmpl.render(
            plan_name=plan.name,
            model_id=ctx.model_id,
            model_version=ctx.model_version,
            run_id=ctx.run_id,
            eval_timestamp=ctx.eval_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            period_start=ctx.period_start.strftime("%Y-%m-%d") if ctx.period_start else None,
            period_end=ctx.period_end.strftime("%Y-%m-%d") if ctx.period_end else None,
            n_current=ctx.n_current,
            n_reference=ctx.n_reference,
            fired_alerts=fired_alerts,
            metric_rows=metric_rows,
            error_rows=error_rows,
            profile_stat_names=profile_stat_names,
            profile_rows=profile_rows,
            chart_html=chart_html,
            scripts=self._charts.include_scripts(),
            style=_BASE_STYLE,
        )

    def render_history(self, reports: list[MonitoringReport]) -> str:
        """Render a history of reports as a time-series dashboard HTML page.

        Reports are sorted internally by ``context.eval_timestamp`` (oldest
        first).  One time-series chart is generated per distinct metric name,
        showing value over time with a dashed threshold reference line when
        the metric spec includes a scalar threshold.

        Args:
            reports: List of ``MonitoringReport`` objects to visualise.
                Must share the same ``model_id`` and ``model_version``.

        Returns:
            Complete HTML document as a string.

        Raises:
            ValueError: If *reports* is empty.
        """
        if not reports:
            raise ValueError("render_history() requires at least one MonitoringReport.")

        sorted_reports = sorted(reports, key=lambda r: r.context.eval_timestamp)
        ctx0 = sorted_reports[0].context
        ctx_last = sorted_reports[-1].context
        plan = sorted_reports[0].plan

        # Collect per-metric time series
        metric_ts: dict[str, dict[str, Any]] = {}  # metric_name → {timestamps, values, threshold}
        for report in sorted_reports:
            ts_str = report.context.eval_timestamp.strftime("%Y-%m-%d %H:%M")
            for r in report.results:
                key = f"{r.spec.name}|{r.spec.feature_name or ''}"
                if key not in metric_ts:
                    display = r.spec.name if not r.spec.feature_name else f"{r.spec.name} ({r.spec.feature_name})"
                    threshold: float | None = None
                    if isinstance(r.spec.threshold, int | float):
                        threshold = float(r.spec.threshold)
                    metric_ts[key] = {
                        "display": display,
                        "timestamps": [],
                        "values": [],
                        "threshold": threshold,
                        "latest": r,
                    }
                if isinstance(r.value, int | float):
                    metric_ts[key]["timestamps"].append(ts_str)
                    metric_ts[key]["values"].append(float(r.value))
                metric_ts[key]["latest"] = r

        timeseries_charts = []
        for entry in metric_ts.values():
            if entry["timestamps"]:
                chart = self._charts.timeseries(
                    timestamps=entry["timestamps"],
                    values=entry["values"],
                    title=entry["display"],
                    threshold=entry["threshold"],
                )
                if chart:
                    timeseries_charts.append(chart)

        # Latest results summary table
        latest_rows = [
            {
                "name": data["latest"].spec.name,
                "feature_name": data["latest"].spec.feature_name or "",
                "value": _fmt_value(data["latest"].value),
                "status_icon": _status_icon(data["latest"].status),
                "row_class": _row_class(data["latest"].status),
                "threshold": str(data["latest"].spec.threshold) if data["latest"].spec.threshold is not None else "—",
            }
            for data in metric_ts.values()
        ]

        tmpl = Template(_HISTORY_TEMPLATE)
        return tmpl.render(
            plan_name=plan.name,
            model_id=ctx0.model_id,
            model_version=ctx0.model_version,
            n_runs=len(sorted_reports),
            first_ts=ctx0.eval_timestamp.strftime("%Y-%m-%d %H:%M"),
            last_ts=ctx_last.eval_timestamp.strftime("%Y-%m-%d %H:%M"),
            timeseries_charts=timeseries_charts,
            latest_rows=latest_rows,
            scripts=self._charts.include_scripts(),
            style=_BASE_STYLE,
        )
