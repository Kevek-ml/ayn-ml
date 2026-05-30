"""No-op chart backend for headless or offline environments.

``NoChartBackend`` satisfies the ``ChartBackend`` protocol by returning empty
strings for all chart methods.  Use it when:

- Rendering in a headless CI environment without internet access.
- Generating reports that will be sent as plain-text emails.
- Embedding the renderer in a pipeline where chart rendering is too slow.

Example::

    renderer = HtmlRenderer(charts=NoChartBackend())
    html = renderer.render(report)   # metric table only, no charts
"""

from __future__ import annotations


class NoChartBackend:
    """ChartBackend that produces no charts.

    All methods return an empty string.  The HTML renderer gracefully omits
    the chart sections when they are empty.

    Example::

        from ayn_ml.renderers import HtmlRenderer, NoChartBackend
        renderer = HtmlRenderer(charts=NoChartBackend())
    """

    def include_scripts(self) -> str:
        """Return empty string — no scripts required.

        Returns:
            Empty string.
        """
        return ""

    def bar_chart(self, labels: list[str], values: list[float], title: str) -> str:
        """Return empty string — no chart rendered.

        Args:
            labels: Unused.
            values: Unused.
            title: Unused.

        Returns:
            Empty string.
        """
        return ""

    def timeseries(
        self,
        timestamps: list[str],
        values: list[float],
        title: str,
        threshold: float | None = None,
    ) -> str:
        """Return empty string — no chart rendered.

        Args:
            timestamps: Unused.
            values: Unused.
            title: Unused.
            threshold: Unused.

        Returns:
            Empty string.
        """
        return ""
