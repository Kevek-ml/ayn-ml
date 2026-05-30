# API Reference

Complete auto-generated reference for all public symbols in ayn-ml.
Docstrings follow the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

| Module | Contents |
|---|---|
| [Core](core.md) | `MonitoringPlan`, `MetricSpec`, schemas, `MetricResult`, `MonitoringReport`, alerts |
| [Metrics](metrics.md) | Registry, `@register_metric`, built-in metric list |
| [Advisor](advisor.md) | `MetricAdvisor`, `SuggestedPlan` |
| [Runner](runner.md) | `Runner` |
| [Data](data.md) | `DataFrameSource`, `SamplingStrategy`, `DataPartitioner` |
| [Stores](stores.md) | `InMemoryStore`, `SqliteStore` |
| [Sinks](sinks.md) | `ResultSink`, `EmailChannel`, `WebhookChannel` |
| [Renderers](renderers.md) | `HtmlRenderer` |

---

!!! note "Public API surface"
    Only symbols in `__all__` of each module are documented here.
    Internal helpers (prefixed `_`) are excluded.
