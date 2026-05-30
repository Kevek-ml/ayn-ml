"""Renderer and chart-backend protocols for HTML report generation.

Two distinct protocols are defined here:

- ``ChartBackend`` — generates individual chart HTML fragments (bar charts,
  time series).  ``PlotlyBackend`` is the default; ``NoChartBackend`` is a
  headless no-op for environments without a display or internet access.

- ``ReportRenderer`` — produces a complete HTML document from one or more
  ``MonitoringReport`` objects.  ``HtmlRenderer`` is the only built-in
  implementation.

Both protocols use structural subtyping — no ABC registration required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ayn_ml.core.result import MonitoringReport


class ChartBackend(Protocol):
    """Protocol for pluggable chart generation within an HTML renderer.

    All methods return a self-contained HTML fragment (a ``<div>`` element
    or empty string) that can be embedded directly in the renderer's output.
    The fragment must NOT include ``<html>``, ``<head>``, or ``<body>`` tags.

    Implementations must also provide ``include_scripts()`` so the renderer
    can inject any required JavaScript once at the top of the page.
    """

    def include_scripts(self) -> str:
        """Return ``<script>`` tags to inject once in the HTML ``<head>``.

        Returns:
            HTML string, or empty string when no scripts are required.
        """
        ...

    def bar_chart(self, labels: list[str], values: list[float], title: str) -> str:
        """Generate a bar chart HTML fragment.

        Args:
            labels: Category labels (x-axis).
            values: Numeric values (y-axis), parallel to *labels*.
            title: Chart title.

        Returns:
            HTML ``<div>`` fragment containing the chart, or empty string.
        """
        ...

    def timeseries(
        self,
        timestamps: list[str],
        values: list[float],
        title: str,
        threshold: float | None = None,
    ) -> str:
        """Generate a time-series line chart HTML fragment.

        Args:
            timestamps: ISO-format timestamp strings (x-axis), oldest first.
            values: Metric values (y-axis), parallel to *timestamps*.
            title: Chart title.
            threshold: Optional horizontal threshold line to overlay.

        Returns:
            HTML ``<div>`` fragment containing the chart, or empty string.
        """
        ...


class ReportRenderer(Protocol):
    """Protocol for HTML report generation from ``MonitoringReport`` objects.

    Implementations receive one or more ``MonitoringReport`` objects and
    return a complete, self-contained HTML document as a string.
    """

    def render(self, report: MonitoringReport) -> str:
        """Render a single monitoring report snapshot as HTML.

        Args:
            report: The completed ``MonitoringReport`` from a Runner run.

        Returns:
            Complete HTML document string.
        """
        ...

    def render_history(self, reports: list[MonitoringReport]) -> str:
        """Render a history of reports as a time-series dashboard HTML page.

        Reports are expected to share the same ``model_id`` and
        ``model_version``.  The renderer sorts them by
        ``context.eval_timestamp`` (oldest first) internally.

        Args:
            reports: List of ``MonitoringReport`` objects to visualise.
                Must contain at least one report.

        Returns:
            Complete HTML document string.

        Raises:
            ValueError: If *reports* is empty.
        """
        ...
