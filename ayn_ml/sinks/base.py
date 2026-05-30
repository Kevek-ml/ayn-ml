"""ResultSink protocol: write-only interface for notification channels.

A ResultSink receives a MonitoringReport after each Runner execution and
dispatches it to an external destination (email, webhook, Slack, etc.).
Sinks are used for alerts and notifications — not for persistence.
Persistence is handled by ResultStore (ayn_ml.stores).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ayn_ml.core.result import MonitoringReport


class ResultSink(Protocol):
    """Write-only interface for notification channels.

    Implementations must be stateless between calls — each ``write()``
    is an independent dispatch.  Errors raised by ``write()`` are caught
    by the Runner and logged as warnings; they do not abort the run.
    """

    def write(self, report: MonitoringReport) -> None:
        """Dispatch a monitoring report to the sink destination.

        Any exception raised by this method is caught and logged by the Runner
        as a warning — it does not abort the run.

        Args:
            report: The completed MonitoringReport from a Runner execution.
        """
        ...
