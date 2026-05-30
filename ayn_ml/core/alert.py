"""Alert policies and rules for threshold-based monitoring notifications.

AlertPolicy defines the contract that any policy must satisfy — a single
``evaluate(result) -> bool`` call that decides whether an alert fires.

ThresholdPolicy is the built-in stateless implementation: it fires when a
MetricResult's value crosses a given threshold (or when the metric's own
pre-computed status indicates a breach, when no custom threshold is set).

AlertRule binds a metric name to a policy and a list of notification
channels (ResultSink implementations).  The Runner evaluates all rules
after metric computation and dispatches channels for every rule that fires.

Policies requiring historical context (ChangePolicy, ConsecutivePolicy)
are not yet implemented — they require store injection into the runner's
alert evaluation path and are deferred to a future release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ayn_ml.core.result import MetricResult
    from ayn_ml.sinks.base import ResultSink


@runtime_checkable
class AlertPolicy(Protocol):
    """Protocol satisfied by all alert policies.

    An AlertPolicy is a stateless callable object: given a single
    ``MetricResult`` it returns ``True`` when the alert should fire.

    Implementations must also expose:

    - ``policy_type`` — short string identifier (e.g. ``"threshold"``).
    - ``details(result)`` — dict of policy-specific context attached to
      the ``FiredAlert`` record.

    The Runner calls ``evaluate()`` once per (AlertRule, MetricResult) pair.
    """

    @property
    def policy_type(self) -> str:
        """Short string identifying the policy kind (e.g. ``"threshold"``)."""
        ...

    def evaluate(self, result: MetricResult) -> bool:
        """Return True when the alert condition is met for *result*."""
        ...

    def details(self, result: MetricResult) -> dict[str, Any]:
        """Return policy-specific context for the FiredAlert record."""
        ...


@dataclass(frozen=True)
class ThresholdPolicy:
    """Stateless threshold-based alert policy.

    Fires when a ``MetricResult``'s value crosses the configured threshold.
    When ``threshold`` is ``None``, delegates to the result's pre-computed
    ``status`` field — i.e., fires whenever ``status is False`` (meaning
    the metric's own spec threshold was breached).

    This is the default policy for simple "alert me when the metric is bad"
    use cases.  It requires no historical data and no store access.

    Args:
        threshold: Custom alert threshold.  When ``None`` (default), fires
            when ``MetricResult.status is False``.  When set, compares the
            result's ``value`` directly, independently of the spec threshold.
        upper_bound: Direction of the custom threshold check.  When ``True``
            (default), fires when ``value > threshold``.  When ``False``,
            fires when ``value < threshold``.  Ignored when ``threshold``
            is ``None``.

    Example::

        # Fire when the metric's own threshold is breached
        policy = ThresholdPolicy()

        # Fire when accuracy drops below 0.80 (regardless of spec)
        policy = ThresholdPolicy(threshold=0.80, upper_bound=False)

        # Fire when PSI exceeds 0.2
        policy = ThresholdPolicy(threshold=0.2, upper_bound=True)
    """

    threshold: float | None = None
    upper_bound: bool = True

    @property
    def policy_type(self) -> str:
        """Policy type identifier."""
        return "threshold"

    def evaluate(self, result: MetricResult) -> bool:
        """Return True when the alert condition is met.

        Args:
            result: MetricResult to evaluate.

        Returns:
            ``True`` when the alert should fire, ``False`` otherwise.
        """
        if self.threshold is not None:
            val = result.value
            if val is None or not isinstance(val, int | float):
                return False
            v = float(val)
            return v > self.threshold if self.upper_bound else v < self.threshold
        # No custom threshold — delegate to pre-computed status.
        return result.status is False

    def details(self, result: MetricResult) -> dict[str, Any]:
        """Return alert detail dict for the FiredAlert record.

        Args:
            result: MetricResult that triggered the alert.

        Returns:
            Dict with ``value``, ``threshold``, and ``upper_bound`` keys.
        """
        used_threshold: float | list[float] | None
        if self.threshold is not None:
            used_threshold = self.threshold
        else:
            used_threshold = result.spec.threshold
        return {
            "value": result.value,
            "threshold": used_threshold,
            "upper_bound": self.upper_bound,
        }


@dataclass
class AlertRule:
    """Binding between a metric, an alert policy, and notification channels.

    When the Runner evaluates ``alert_rules``, it matches each rule's
    ``metric_name`` to a computed ``MetricResult`` and calls
    ``policy.evaluate(result)``.  If the policy fires, every channel in
    ``channels`` receives ``write(report)`` with the completed
    ``MonitoringReport``.

    Args:
        metric_name: Registry name of the metric to watch (must match a
            ``MetricSpec.name`` in the monitoring plan).
        policy: Alert policy that decides whether the alert fires.
        channels: List of ``ResultSink`` implementations to notify when
            the alert fires.  Typically ``[EmailChannel(...)]``,
            ``[WebhookChannel(...)]``, or a mix of both.

    Example::

        from ayn_ml.core.alert import AlertRule, ThresholdPolicy
        from ayn_ml.sinks.email import EmailChannel

        channel = EmailChannel(host="smtp.example.com", to_addrs=["ops@example.com"])
        rule = AlertRule(
            metric_name="psi",
            policy=ThresholdPolicy(threshold=0.2),
            channels=[channel],
        )
        runner.run(plan, df, alert_rules=[rule])
    """

    metric_name: str
    policy: AlertPolicy  # type: ignore[type-arg]
    channels: list[ResultSink] = field(default_factory=list)  # type: ignore[type-arg]
