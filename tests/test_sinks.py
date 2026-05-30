"""Tests for ayn_ml.sinks — EmailChannel and WebhookChannel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ayn_ml.core.result import ExecutionContext, FiredAlert, MetricError, MetricResult, MonitoringReport
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MonitoringPlan
from ayn_ml.sinks.email import EmailChannel, _plain_text_summary
from ayn_ml.sinks.webhook import WebhookChannel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _report(
    plan_name: str = "test_plan",
    model_id: str = "model_x",
    model_version: str = "1.0",
    results: list[MetricResult] | None = None,
    fired_alerts: list[FiredAlert] | None = None,
    errors: list[MetricError] | None = None,
) -> MonitoringReport:
    plan = MonitoringPlan(
        name=plan_name,
        model_id=model_id,
        model_version=model_version,
        data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
        metrics=[MetricSpec(name="accuracy")],
    )
    ctx = ExecutionContext(
        model_id=model_id,
        model_version=model_version,
        eval_timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        n_current=100,
    )
    return MonitoringReport(
        plan=plan,
        context=ctx,
        results=results or [],
        errors=errors or [],
        fired_alerts=fired_alerts or [],
        profile=None,
    )


def _result_pass() -> MetricResult:
    return MetricResult(spec=MetricSpec(name="accuracy", threshold=0.8), value=0.92, status=True)


def _result_fail() -> MetricResult:
    return MetricResult(spec=MetricSpec(name="psi", threshold=0.1), value=0.15, status=False)


# ---------------------------------------------------------------------------
# WebhookChannel
# ---------------------------------------------------------------------------


class TestWebhookChannel:
    def _make_channel(self, url: str = "https://hooks.example.com/test") -> WebhookChannel:
        return WebhookChannel(url=url)

    def _mock_response(self, status: int = 200) -> MagicMock:
        response = MagicMock()
        response.status = status
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)
        return response

    def test_posts_json_to_url(self):
        channel = self._make_channel()
        report = _report(results=[_result_pass()])
        mock_resp = self._mock_response()

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            channel.write(report)

        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        assert req.full_url == "https://hooks.example.com/test"
        assert req.get_method() == "POST"

    def test_content_type_is_json(self):
        channel = self._make_channel()
        report = _report()
        mock_resp = self._mock_response()

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            channel.write(report)

        req = mock_open.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"

    def test_payload_is_valid_json(self):
        channel = self._make_channel()
        report = _report(results=[_result_pass(), _result_fail()])
        mock_resp = self._mock_response()
        captured: list[bytes] = []

        def capture_request(req, timeout=None):
            captured.append(req.data)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=capture_request):
            channel.write(report)

        assert len(captured) == 1
        payload = json.loads(captured[0].decode("utf-8"))
        # to_dict() must produce a dict with context info
        assert "context" in payload or "run_id" in str(payload)

    def test_extra_headers_are_sent(self):
        channel = WebhookChannel(
            url="https://hooks.example.com/test",
            extra_headers={"Authorization": "Bearer tok", "X-Source": "ayn-ml"},
        )
        report = _report()
        mock_resp = self._mock_response()

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            channel.write(report)

        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer tok"
        assert req.get_header("X-source") == "ayn-ml"

    def test_timeout_passed_to_urlopen(self):
        channel = WebhookChannel(url="https://hooks.example.com/test", timeout=5)
        report = _report()
        mock_resp = self._mock_response()

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            channel.write(report)

        _, kwargs = mock_open.call_args
        assert kwargs.get("timeout") == 5

    def test_http_error_propagates(self):
        import urllib.error

        channel = self._make_channel()
        report = _report()

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 500, "err", {}, None)):  # type: ignore[arg-type]
            with pytest.raises(urllib.error.HTTPError):
                channel.write(report)

    def test_network_error_propagates(self):
        import urllib.error

        channel = self._make_channel()
        report = _report()

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            with pytest.raises(urllib.error.URLError):
                channel.write(report)


# ---------------------------------------------------------------------------
# EmailChannel — plain text summary
# ---------------------------------------------------------------------------


class TestPlainTextSummary:
    def test_contains_plan_name(self):
        report = _report(plan_name="fraud_monitor")
        text = _plain_text_summary(report)
        assert "fraud_monitor" in text

    def test_contains_model_id(self):
        report = _report(model_id="fraud_v2", model_version="3.1")
        text = _plain_text_summary(report)
        assert "fraud_v2" in text
        assert "3.1" in text

    def test_contains_metric_results(self):
        report = _report(results=[_result_pass(), _result_fail()])
        text = _plain_text_summary(report)
        assert "accuracy" in text
        assert "psi" in text
        assert "PASS" in text
        assert "FAIL" in text

    def test_contains_fired_alerts(self):
        alerts = [FiredAlert(metric_name="psi", policy_type="threshold", details={"value": 0.15})]
        report = _report(fired_alerts=alerts)
        text = _plain_text_summary(report)
        assert "FIRED ALERTS" in text
        assert "psi" in text

    def test_contains_errors(self):
        errors = [MetricError(metric_name="auc", error_type="SchemaError", message="missing col")]
        report = _report(errors=errors)
        text = _plain_text_summary(report)
        assert "ERRORS" in text
        assert "auc" in text
        assert "missing col" in text


# ---------------------------------------------------------------------------
# EmailChannel — SMTP dispatch
# ---------------------------------------------------------------------------


class TestEmailChannel:
    def _channel(self, **kwargs) -> EmailChannel:
        defaults = dict(
            host="smtp.example.com",
            to_addrs=["ops@example.com"],
            from_addr="monitor@example.com",
            port=587,
            use_tls=True,
        )
        defaults.update(kwargs)
        return EmailChannel(**defaults)

    def _mock_smtp(self):
        smtp_instance = MagicMock()
        smtp_instance.__enter__ = lambda s: s
        smtp_instance.__exit__ = MagicMock(return_value=False)
        return smtp_instance

    def test_sendmail_called_with_correct_recipients(self):
        channel = self._channel(to_addrs=["alice@example.com", "bob@example.com"])
        report = _report()
        smtp_inst = self._mock_smtp()

        with patch("smtplib.SMTP", return_value=smtp_inst):
            channel.write(report)

        smtp_inst.sendmail.assert_called_once()
        _, recipients, _ = smtp_inst.sendmail.call_args[0]
        assert recipients == ["alice@example.com", "bob@example.com"]

    def test_starttls_called_when_use_tls_true(self):
        channel = self._channel(use_tls=True)
        report = _report()
        smtp_inst = self._mock_smtp()

        with patch("smtplib.SMTP", return_value=smtp_inst):
            channel.write(report)

        smtp_inst.starttls.assert_called_once()

    def test_starttls_not_called_when_use_tls_false(self):
        channel = self._channel(use_tls=False)
        report = _report()
        smtp_inst = self._mock_smtp()

        with patch("smtplib.SMTP", return_value=smtp_inst):
            channel.write(report)

        smtp_inst.starttls.assert_not_called()

    def test_login_called_when_credentials_provided(self):
        channel = self._channel(username="user@example.com", password="s3cr3t")
        report = _report()
        smtp_inst = self._mock_smtp()

        with patch("smtplib.SMTP", return_value=smtp_inst):
            channel.write(report)

        smtp_inst.login.assert_called_once_with("user@example.com", "s3cr3t")

    def test_login_not_called_when_no_credentials(self):
        channel = self._channel(username="", password="")
        report = _report()
        smtp_inst = self._mock_smtp()

        with patch("smtplib.SMTP", return_value=smtp_inst):
            channel.write(report)

        smtp_inst.login.assert_not_called()

    def test_subject_contains_plan_name(self):
        import email
        import email.header

        channel = self._channel()
        report = _report(plan_name="churn_v2")
        smtp_inst = self._mock_smtp()
        sent_messages: list[str] = []

        def capture_sendmail(from_addr, to_addrs, message):
            sent_messages.append(message)

        smtp_inst.sendmail.side_effect = capture_sendmail

        with patch("smtplib.SMTP", return_value=smtp_inst):
            channel.write(report)

        assert sent_messages
        # Subject may be RFC 2047-encoded (e.g. =?utf-8?b?...?=) — decode before checking
        parsed = email.message_from_string(sent_messages[0])
        raw_subject = parsed.get("Subject", "")
        decoded_parts = email.header.decode_header(raw_subject)
        subject_text = "".join(
            part.decode(enc or "utf-8") if isinstance(part, bytes) else part for part, enc in decoded_parts
        )
        assert "churn_v2" in subject_text

    def test_smtp_error_propagates(self):
        import smtplib

        channel = self._channel()
        report = _report()
        smtp_inst = self._mock_smtp()
        smtp_inst.sendmail.side_effect = smtplib.SMTPException("connection refused")

        with patch("smtplib.SMTP", return_value=smtp_inst):
            with pytest.raises(smtplib.SMTPException):
                channel.write(report)

    def test_empty_to_addrs_raises_on_init(self):
        with pytest.raises(ValueError, match="to_addrs"):
            EmailChannel(host="smtp.example.com", to_addrs=[])

    def test_with_html_renderer_sends_multipart(self):
        from ayn_ml.renderers.html import HtmlRenderer
        from ayn_ml.renderers.no_chart import NoChartBackend

        renderer = HtmlRenderer(charts=NoChartBackend())
        channel = self._channel(renderer=renderer)
        report = _report(results=[_result_pass()])
        smtp_inst = self._mock_smtp()
        sent_messages: list[str] = []

        def capture_sendmail(from_addr, to_addrs, message):
            sent_messages.append(message)

        smtp_inst.sendmail.side_effect = capture_sendmail

        with patch("smtplib.SMTP", return_value=smtp_inst):
            channel.write(report)

        assert sent_messages
        # Multipart message should contain both text/plain and text/html
        msg_str = sent_messages[0]
        assert "text/plain" in msg_str
        assert "text/html" in msg_str
