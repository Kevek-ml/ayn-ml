"""EmailChannel: SMTP notification sink for monitoring reports.

``EmailChannel`` sends a monitoring report summary via SMTP using only
Python stdlib (``smtplib``, ``email``).  No third-party mail library is
required.

The channel satisfies the ``ResultSink`` protocol.  By default it sends a
plain-text summary of the report.  When an ``HtmlRenderer`` is supplied via
``renderer``, it sends a multipart email with both HTML and plain-text parts
(the plain-text part serves as a fallback for mail clients that do not render
HTML).

Example::

    from ayn_ml.sinks.email import EmailChannel
    from ayn_ml.core.alert import AlertRule, ThresholdPolicy

    channel = EmailChannel(
        host="smtp.example.com",
        to_addrs=["ops@example.com"],
        from_addr="monitoring@example.com",
        username="monitoring@example.com",
        password="s3cr3t",
        port=587,
        use_tls=True,
    )
    rule = AlertRule(
        metric_name="accuracy",
        policy=ThresholdPolicy(threshold=0.80, upper_bound=False),
        channels=[channel],
    )
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ayn_ml.core.result import MonitoringReport
    from ayn_ml.renderers.html import HtmlRenderer

_log = logging.getLogger(__name__)


def _plain_text_summary(report: MonitoringReport) -> str:
    """Build a plain-text summary of a MonitoringReport.

    Args:
        report: Completed MonitoringReport.

    Returns:
        Plain-text string suitable for email body.
    """
    ctx = report.context
    lines: list[str] = [
        f"ayn-ml Monitoring Report — {report.plan.name}",
        "=" * 60,
        f"Model   : {ctx.model_id} v{ctx.model_version}",
        f"Run ID  : {ctx.run_id}",
        f"Time    : {ctx.eval_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Rows    : {ctx.n_current}" + (f" / {ctx.n_reference} ref" if ctx.n_reference else ""),
        "",
    ]

    if report.fired_alerts:
        lines.append(f"FIRED ALERTS ({len(report.fired_alerts)})")
        lines.append("-" * 40)
        for fa in report.fired_alerts:
            detail_str = ", ".join(f"{k}={v}" for k, v in fa.details.items())
            lines.append(f"  • {fa.metric_name} [{fa.policy_type}]: {detail_str}")
        lines.append("")

    lines.append("METRIC RESULTS")
    lines.append("-" * 40)
    for r in report.results:
        value_str = f"{r.value:.4g}" if isinstance(r.value, int | float) else str(r.value or "—")
        status_str = "PASS" if r.status is True else ("FAIL" if r.status is False else "N/A")
        feat = f" [{r.spec.feature_name}]" if r.spec.feature_name else ""
        thr = f" (threshold={r.spec.threshold})" if r.spec.threshold is not None else ""
        lines.append(f"  {status_str:4s}  {r.spec.name}{feat}: {value_str}{thr}")

    if report.errors:
        lines.append("")
        lines.append(f"ERRORS ({len(report.errors)})")
        lines.append("-" * 40)
        for e in report.errors:
            lines.append(f"  • {e.metric_name} [{e.error_type}]: {e.message}")

    return "\n".join(lines)


class EmailChannel:
    """ResultSink that sends a MonitoringReport summary via SMTP.

    Uses Python's built-in ``smtplib`` and ``email`` libraries — no
    third-party mail library is required.

    When ``renderer`` is provided, the email is multipart (HTML + plain
    text fallback).  Otherwise only a plain-text body is sent.

    Args:
        host: SMTP server hostname (e.g. ``"smtp.gmail.com"``).
        to_addrs: List of recipient email addresses.
        from_addr: Sender email address.  Defaults to the first element
            of *to_addrs* when not set.
        port: SMTP port.  587 for STARTTLS (default), 465 for SSL,
            25 for unencrypted.
        username: SMTP authentication username.  Required by most
            cloud mail providers.
        password: SMTP authentication password or app-specific token.
        subject_template: Python format string for the email subject.
            Available keys: ``{plan_name}``, ``{model_id}``,
            ``{model_version}``, ``{n_alerts}``.
        use_tls: When ``True`` (default), use STARTTLS.  Set to
            ``False`` for SSL-wrapped connections (port 465) or for
            local development SMTP servers.
        timeout: SMTP connection timeout in seconds.  Defaults to 30.
        renderer: Optional ``HtmlRenderer`` instance.  When supplied,
            sends a multipart email with an HTML body plus a plain-text
            fallback.

    Example::

        channel = EmailChannel(
            host="smtp.gmail.com",
            to_addrs=["ops@example.com"],
            from_addr="monitoring@example.com",
            username="monitoring@example.com",
            password="app-password",
        )
    """

    def __init__(
        self,
        host: str,
        to_addrs: list[str],
        from_addr: str = "",
        port: int = 587,
        username: str = "",
        password: str = "",
        subject_template: str = "[ayn-ml] {plan_name} — {n_alerts} alert(s) fired",
        use_tls: bool = True,
        timeout: int = 30,
        renderer: HtmlRenderer | None = None,
    ) -> None:
        """Initialise the email channel.

        Args:
            host: SMTP server hostname.
            to_addrs: Recipient email addresses.
            from_addr: Sender address.  Defaults to ``to_addrs[0]``.
            port: SMTP port (default 587 for STARTTLS).
            username: SMTP username.
            password: SMTP password or app token.
            subject_template: Email subject format string.
            use_tls: Whether to use STARTTLS.
            timeout: Connection timeout in seconds.
            renderer: Optional HtmlRenderer for HTML email bodies.
        """
        if not to_addrs:
            raise ValueError("EmailChannel: to_addrs must contain at least one address.")
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from_addr = from_addr or to_addrs[0]
        self._to_addrs = list(to_addrs)
        self._subject_template = subject_template
        self._use_tls = use_tls
        self._timeout = timeout
        self._renderer = renderer

    def _build_subject(self, report: MonitoringReport) -> str:
        """Render the email subject from the template.

        Args:
            report: MonitoringReport to extract context from.

        Returns:
            Rendered subject string.
        """
        ctx = report.context
        kwargs: dict[str, Any] = {
            "plan_name": report.plan.name,
            "model_id": ctx.model_id,
            "model_version": ctx.model_version,
            "n_alerts": len(report.fired_alerts),
        }
        try:
            return self._subject_template.format(**kwargs)
        except KeyError:
            _log.warning("EmailChannel: invalid subject_template key; using default subject.")
            return f"[ayn-ml] {report.plan.name} — monitoring report"

    def write(self, report: MonitoringReport) -> None:
        """Send the MonitoringReport summary via SMTP.

        Args:
            report: Completed ``MonitoringReport`` from a Runner execution.

        Raises:
            smtplib.SMTPException: On SMTP protocol errors.
            OSError: On network failures.
        """
        subject = self._build_subject(report)
        plain = _plain_text_summary(report)

        if self._renderer is not None:
            msg: MIMEMultipart | MIMEText = MIMEMultipart("alternative")
            msg.attach(MIMEText(plain, "plain", "utf-8"))
            try:
                html_body = self._renderer.render(report)
                msg.attach(MIMEText(html_body, "html", "utf-8"))
            except Exception as exc:  # noqa: BLE001
                _log.warning("EmailChannel: renderer.render() failed (%s); sending plain text only.", exc)
        else:
            msg = MIMEText(plain, "plain", "utf-8")

        msg["Subject"] = subject
        msg["From"] = self._from_addr
        msg["To"] = ", ".join(self._to_addrs)

        _log.debug(
            "EmailChannel: sending to %s via %s:%s (run_id=%s)",
            self._to_addrs,
            self._host,
            self._port,
            report.context.run_id,
        )

        smtp_cls = smtplib.SMTP
        with smtp_cls(self._host, self._port, timeout=self._timeout) as smtp:
            if self._use_tls:
                smtp.starttls()
            if self._username:
                smtp.login(self._username, self._password)
            smtp.sendmail(self._from_addr, self._to_addrs, msg.as_string())

        _log.info(
            "EmailChannel: sent '%s' to %s (run_id=%s)",
            subject,
            self._to_addrs,
            report.context.run_id,
        )
