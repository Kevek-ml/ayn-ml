"""Write-only notification sinks: email and webhook channels.

Public API::

    from ayn_ml.sinks import EmailChannel, WebhookChannel, ResultSink

    # Webhook — POST JSON to an HTTP endpoint
    channel = WebhookChannel(url="https://hooks.example.com/monitoring")

    # Email — send via SMTP (stdlib only)
    channel = EmailChannel(host="smtp.example.com", to_addrs=["ops@example.com"])
"""

from typing import Any

from ayn_ml.sinks.base import ResultSink
from ayn_ml.sinks.email import EmailChannel
from ayn_ml.sinks.webhook import WebhookChannel

# Populated at import time by an installed extension; None otherwise.
# Accessing a None value raises TypeError with a clear message at call time.
SlackChannel: type[Any] | None = None

__all__ = ["EmailChannel", "ResultSink", "SlackChannel", "WebhookChannel"]
