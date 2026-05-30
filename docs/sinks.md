# Sinks — Slack Notifications

> **Status:** API and class structure are defined. `SlackChannel.send()` is not yet implemented.

## SlackChannel

**Import:** `from ayn_ml_pro.sinks import SlackChannel`

**Extra required:** `pip install "ayn-ml-pro[slack]"`

> **Stub in `ayn-ml`:** `ayn_ml.sinks` exports `SlackChannel: type[Any] | None = None`. This stub is populated by `ayn_ml_pro._extension` when `ayn-ml-pro` is installed; it is `None` otherwise. Import from `ayn_ml_pro.sinks` directly in all production code.

`SlackChannel` implements the `ResultSink` protocol from `ayn-ml` (Apache 2.0). Pass it to any `Runner` and it will post a summary of each `MonitoringReport` to a Slack channel via the Slack Web API.

### Constructor

```python
SlackChannel(
    token: str,           # Slack bot token — xoxb-...
    channel: str,         # Channel name or ID — "#ml-alerts" or "C012AB3CD"
    mention: str | None,  # Optional mention prepended to each alert — "@here" or "<@U123456>"
)
```

Credentials must not be hardcoded. Use an environment variable:

```python
import os
from ayn_ml_pro.sinks import SlackChannel

sink = SlackChannel(
    token=os.environ["SLACK_BOT_TOKEN"],
    channel="#ml-alerts",
    mention="@here",
)
```

### Wiring to a Runner

```python
from ayn_ml.runner import Runner
from ayn_ml_pro.sinks import SlackChannel
from ayn_ml_pro.stores import S3Store

store = S3Store(bucket="my-monitoring-bucket")
sink = SlackChannel(token=os.environ["SLACK_BOT_TOKEN"], channel="#ml-alerts")

runner = Runner(plan, store=store, sinks=[sink])
runner.run(df_current, ref=df_reference)
```

### Error handling

`SlackChannel.send()` raises `SlackDeliveryError` (from `ayn_ml_pro.exceptions`) if the Slack API call fails. The `CloudRunner` catches per-sink errors and records them in `MonitoringReport.errors` — a delivery failure never stops the run.

### Required Slack bot scopes

The bot token must have the `chat:write` scope. If you use `@here` or `@channel` in `mention`, the bot also needs `chat:write.public` for public channels.

---

## Exception reference

| Exception | When raised |
|-----------|-------------|
| `SlackDeliveryError` | Slack API call failed (network error, invalid token, channel not found) |
| `ImportError` | `slack-sdk` not installed — install with `pip install "ayn-ml-pro[slack]"` |
