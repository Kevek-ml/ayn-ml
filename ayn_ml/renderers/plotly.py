"""Plotly-based chart backend for interactive HTML report charts.

``PlotlyBackend`` generates interactive Plotly charts as self-contained HTML
fragments using ``plotly.graph_objects``.  Charts are rendered inline in the
HTML report with a single Plotly CDN script tag injected once by the renderer.

Plotly is a core dependency of ayn-ml (not optional) so no import guard is
needed here.

Example::

    from ayn_ml.renderers import HtmlRenderer, PlotlyBackend
    renderer = HtmlRenderer(charts=PlotlyBackend())   # default
    html = renderer.render(report)
"""

from __future__ import annotations

_PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

# Colour palette for status indicators
_PASS_COLOR = "#28a745"
_FAIL_COLOR = "#dc3545"
_NEUTRAL_COLOR = "#6c757d"


class PlotlyBackend:
    """ChartBackend that renders interactive Plotly charts.

    Charts are returned as ``<div>`` HTML fragments (``full_html=False``,
    ``include_plotlyjs=False``).  The ``include_scripts()`` method returns
    the Plotly CDN ``<script>`` tag so the renderer can inject it once per
    HTML page.

    Example::

        from ayn_ml.renderers import HtmlRenderer, PlotlyBackend
        renderer = HtmlRenderer(charts=PlotlyBackend())
    """

    def include_scripts(self) -> str:
        """Return the Plotly CDN script tag.

        Returns:
            ``<script>`` tag string pointing to Plotly CDN.
        """
        return _PLOTLY_CDN

    def bar_chart(self, labels: list[str], values: list[float], title: str) -> str:
        """Generate an interactive bar chart as an HTML fragment.

        Args:
            labels: Category labels for the x-axis.
            values: Numeric values for the y-axis, parallel to *labels*.
            title: Chart title displayed above the plot.

        Returns:
            HTML ``<div>`` fragment containing the Plotly chart.
            Empty string if *labels* or *values* are empty.
        """
        if not labels or not values:
            return ""

        import plotly.graph_objects as go
        import plotly.io as pio

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=_NEUTRAL_COLOR,
                text=[f"{v:.4g}" for v in values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=title,
            margin={"l": 40, "r": 20, "t": 40, "b": 40},
            height=350,
            plot_bgcolor="white",
            yaxis={"gridcolor": "#e0e0e0"},
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    def timeseries(
        self,
        timestamps: list[str],
        values: list[float],
        title: str,
        threshold: float | None = None,
    ) -> str:
        """Generate an interactive time-series line chart as an HTML fragment.

        Args:
            timestamps: ISO-format timestamp strings (x-axis), oldest first.
            values: Metric values (y-axis), parallel to *timestamps*.
            title: Chart title displayed above the plot.
            threshold: Optional scalar threshold to draw as a horizontal
                dashed reference line.

        Returns:
            HTML ``<div>`` fragment containing the Plotly chart.
            Empty string if *timestamps* or *values* are empty.
        """
        if not timestamps or not values:
            return ""

        import plotly.graph_objects as go
        import plotly.io as pio

        traces: list[go.BaseTraceType] = [
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                name=title,
                line={"color": _NEUTRAL_COLOR, "width": 2},
                marker={"size": 6},
            )
        ]

        if threshold is not None:
            traces.append(
                go.Scatter(
                    x=[timestamps[0], timestamps[-1]],
                    y=[threshold, threshold],
                    mode="lines",
                    name="Threshold",
                    line={"color": _FAIL_COLOR, "width": 1, "dash": "dash"},
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            margin={"l": 40, "r": 20, "t": 40, "b": 40},
            height=350,
            plot_bgcolor="white",
            yaxis={"gridcolor": "#e0e0e0"},
            xaxis={"gridcolor": "#e0e0e0"},
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)
