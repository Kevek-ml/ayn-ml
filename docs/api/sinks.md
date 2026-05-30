# Sinks

Result sinks fire notifications when alerts are triggered.
They implement the `ResultSink` protocol and receive only the reports that
contain fired alerts.

---

## Protocol

::: ayn_ml.sinks.base.ResultSink

---

## EmailChannel

::: ayn_ml.sinks.email.EmailChannel

---

## WebhookChannel

::: ayn_ml.sinks.webhook.WebhookChannel

---

A Slack notification channel is available in the commercial edition.
