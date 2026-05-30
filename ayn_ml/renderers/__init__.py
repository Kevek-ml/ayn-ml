"""HTML rendering for MonitoringReport objects.

Public API::

    from ayn_ml.renderers import HtmlRenderer, PlotlyBackend, NoChartBackend

    # Default — interactive Plotly charts
    html = HtmlRenderer().render(report)

    # Tables only (headless / offline)
    html = HtmlRenderer(charts=NoChartBackend()).render(report)

    # Time-series dashboard across multiple runs
    html = HtmlRenderer().render_history([report1, report2, report3])
"""

from ayn_ml.renderers.base import ChartBackend, ReportRenderer
from ayn_ml.renderers.html import HtmlRenderer
from ayn_ml.renderers.no_chart import NoChartBackend
from ayn_ml.renderers.plotly import PlotlyBackend

__all__ = [
    "ChartBackend",
    "HtmlRenderer",
    "NoChartBackend",
    "PlotlyBackend",
    "ReportRenderer",
]
