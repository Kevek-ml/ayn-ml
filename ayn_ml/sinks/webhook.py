"""WebhookChannel: HTTP POST notification sink for monitoring reports.

``WebhookChannel`` sends the serialised ``MonitoringReport`` as a JSON
payload to a configurable URL using only Python stdlib (``urllib.request``).
No third-party HTTP library is required.

The channel satisfies the ``ResultSink`` protocol.  It is typically used in
an ``AlertRule`` to push alert data to an external service (incident
management, dashboards, Zapier, etc.).

Example::

    from ayn_ml.sinks.webhook import WebhookChannel
    from ayn_ml.core.alert import AlertRule, ThresholdPolicy

    channel = WebhookChannel(
        url="https://hooks.example.com/monitoring",
        extra_headers={"Authorization": "Bearer my-token"},
        timeout=10,
    )
    rule = AlertRule(
        metric_name="psi",
        policy=ThresholdPolicy(threshold=0.2),
        channels=[channel],
    )
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ayn_ml.core.result import MonitoringReport

_log = logging.getLogger(__name__)


class WebhookChannel:
    """ResultSink that POSTs a MonitoringReport as JSON to an HTTP endpoint.

    Uses Python's built-in ``urllib.request`` — no third-party HTTP library
    required.  The full ``MonitoringReport.to_dict()`` payload is sent with
    ``Content-Type: application/json``.

    Args:
        url: Destination URL.  Must include the scheme
            (e.g. ``"https://hooks.example.com/monitoring"``).
        extra_headers: Additional HTTP headers to include in the request
            (e.g. ``{"Authorization": "Bearer token"}``).  Merged with the
            default ``Content-Type`` header; caller-supplied values take
            precedence.
        timeout: Request timeout in seconds.  Defaults to 30.

    Example::

        channel = WebhookChannel(
            url="https://hooks.example.com/monitoring",
            extra_headers={"X-API-Key": "secret"},
            timeout=10,
        )
    """

    def __init__(
        self,
        url: str,
        extra_headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialise the webhook channel.

        Args:
            url: Destination URL for the HTTP POST request.
            extra_headers: Additional HTTP headers (merged with the default
                ``Content-Type`` header).
            timeout: Request timeout in seconds.
        """
        self._url = url
        self._extra_headers: dict[str, str] = extra_headers or {}
        self._timeout = timeout

    def write(self, report: MonitoringReport) -> None:
        """POST the MonitoringReport as JSON to the configured URL.

        Serialises the report with ``MonitoringReport.to_dict()``.  Raises
        on HTTP errors (4xx/5xx) and network failures; the Runner catches
        and logs these.

        Args:
            report: Completed ``MonitoringReport`` from a Runner execution.

        Raises:
            urllib.error.URLError: On network failures or DNS resolution errors.
            urllib.error.HTTPError: On HTTP 4xx/5xx responses.
        """
        payload = json.dumps(report.to_dict(), default=str).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        headers.update(self._extra_headers)

        req = urllib.request.Request(
            url=self._url,
            data=payload,
            headers=headers,
            method="POST",
        )
        _log.debug(
            "WebhookChannel: POST %s — run_id=%s model=%s",
            self._url,
            report.context.run_id,
            report.context.model_id,
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 — caller-supplied URL; SSRF is caller's responsibility
            status = resp.status
            _log.info(
                "WebhookChannel: POST %s → HTTP %s (run_id=%s)",
                self._url,
                status,
                report.context.run_id,
            )
