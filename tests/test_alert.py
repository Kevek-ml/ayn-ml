"""Tests for ayn_ml.core.alert — AlertPolicy, ThresholdPolicy, AlertRule."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

pd = pytest.importorskip("pandas")

from ayn_ml.core.alert import AlertRule, ThresholdPolicy
from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MonitoringPlan
from ayn_ml.runner import Runner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spec(name: str = "accuracy", threshold: float | None = None, upper_bound: bool = True) -> MetricSpec:
    return MetricSpec(name=name, threshold=threshold, upper_bound=upper_bound)


def _result(value: float, status: bool | None, threshold: float | None = None) -> MetricResult:
    return MetricResult(spec=_spec(threshold=threshold), value=value, status=status)


def _plan(threshold: float | None = None) -> MonitoringPlan:
    return MonitoringPlan(
        name="test",
        model_id="m",
        model_version="1",
        data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
        metrics=[MetricSpec(name="accuracy", threshold=threshold)],
    )


def _df() -> pd.DataFrame:
    return pd.DataFrame({"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 1, 0]})


# ---------------------------------------------------------------------------
# ThresholdPolicy — stateless threshold check
# ---------------------------------------------------------------------------


class TestThresholdPolicyWithCustomThreshold:
    def test_fires_when_value_exceeds_upper_threshold(self):
        policy = ThresholdPolicy(threshold=0.5, upper_bound=True)
        result = _result(value=0.6, status=True)
        assert policy.evaluate(result) is True

    def test_does_not_fire_when_value_below_upper_threshold(self):
        policy = ThresholdPolicy(threshold=0.5, upper_bound=True)
        result = _result(value=0.4, status=True)
        assert policy.evaluate(result) is False

    def test_fires_when_value_below_lower_threshold(self):
        policy = ThresholdPolicy(threshold=0.8, upper_bound=False)
        result = _result(value=0.75, status=False)
        assert policy.evaluate(result) is True

    def test_does_not_fire_when_value_above_lower_threshold(self):
        policy = ThresholdPolicy(threshold=0.8, upper_bound=False)
        result = _result(value=0.85, status=True)
        assert policy.evaluate(result) is False

    def test_does_not_fire_when_value_equals_threshold(self):
        policy = ThresholdPolicy(threshold=0.5, upper_bound=True)
        result = _result(value=0.5, status=None)
        assert policy.evaluate(result) is False

    def test_does_not_fire_for_none_value(self):
        policy = ThresholdPolicy(threshold=0.5)
        result = _result(value=None, status=None)  # type: ignore[arg-type]
        assert policy.evaluate(result) is False

    def test_does_not_fire_for_string_value(self):
        policy = ThresholdPolicy(threshold=0.5)
        result = MetricResult(spec=_spec(), value="high", status=None)
        assert policy.evaluate(result) is False

    def test_integer_value_is_accepted(self):
        policy = ThresholdPolicy(threshold=1, upper_bound=True)
        result = _result(value=2, status=None)
        assert policy.evaluate(result) is True


class TestThresholdPolicyWithoutCustomThreshold:
    """When threshold=None, policy delegates to MetricResult.status."""

    def test_fires_when_status_is_false(self):
        policy = ThresholdPolicy()
        result = _result(value=0.3, status=False)
        assert policy.evaluate(result) is True

    def test_does_not_fire_when_status_is_true(self):
        policy = ThresholdPolicy()
        result = _result(value=0.9, status=True)
        assert policy.evaluate(result) is False

    def test_does_not_fire_when_status_is_none(self):
        policy = ThresholdPolicy()
        result = _result(value=0.5, status=None)
        assert policy.evaluate(result) is False


class TestThresholdPolicyMeta:
    def test_policy_type_is_threshold(self):
        assert ThresholdPolicy().policy_type == "threshold"

    def test_details_with_custom_threshold(self):
        policy = ThresholdPolicy(threshold=0.2, upper_bound=True)
        result = _result(value=0.25, status=False)
        details = policy.details(result)
        assert details["value"] == 0.25
        assert details["threshold"] == 0.2
        assert details["upper_bound"] is True

    def test_details_without_custom_threshold_uses_spec_threshold(self):
        policy = ThresholdPolicy()
        result = MetricResult(
            spec=MetricSpec(name="psi", threshold=0.1),
            value=0.15,
            status=False,
        )
        details = policy.details(result)
        assert details["threshold"] == 0.1
        assert details["value"] == 0.15

    def test_frozen_dataclass(self):
        policy = ThresholdPolicy(threshold=0.5)
        with pytest.raises(Exception):  # frozen=True → AttributeError or FrozenInstanceError
            policy.threshold = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------


class TestAlertRule:
    def test_basic_construction(self):
        policy = ThresholdPolicy(threshold=0.2)
        channel = MagicMock()
        rule = AlertRule(metric_name="psi", policy=policy, channels=[channel])
        assert rule.metric_name == "psi"
        assert rule.policy is policy
        assert rule.channels == [channel]

    def test_default_channels_is_empty_list(self):
        rule = AlertRule(metric_name="accuracy", policy=ThresholdPolicy())
        assert rule.channels == []
        # Each instance gets its own list (no shared mutable default)
        rule2 = AlertRule(metric_name="auc", policy=ThresholdPolicy())
        rule.channels.append(MagicMock())
        assert rule2.channels == []


# ---------------------------------------------------------------------------
# Runner integration — end-to-end alert dispatch
# ---------------------------------------------------------------------------


class TestRunnerAlertIntegration:
    # accuracy=1.0 on test data ([1,0,1,0] vs [1,0,1,0])
    # upper_bound=True → fires when value > threshold → 1.0 > 0.5 → fires ✓

    def test_alert_fires_and_appears_in_report(self):
        """ThresholdPolicy fires → FiredAlert in MonitoringReport."""
        channel = MagicMock()
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.5, upper_bound=True),  # 1.0 > 0.5 → fires
            channels=[channel],
        )
        report = Runner().run(_plan(), _df(), alert_rules=[rule])
        assert len(report.fired_alerts) == 1
        fa = report.fired_alerts[0]
        assert fa.metric_name == "accuracy"
        assert fa.policy_type == "threshold"

    def test_channel_write_called_when_alert_fires(self):
        """Channels on a fired AlertRule receive write(report)."""
        channel = MagicMock()
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.5, upper_bound=True),
            channels=[channel],
        )
        report = Runner().run(_plan(), _df(), alert_rules=[rule])
        channel.write.assert_called_once_with(report)

    def test_channel_not_called_when_alert_does_not_fire(self):
        """Channels are NOT called when the threshold is not breached."""
        channel = MagicMock()
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=1.5, upper_bound=True),  # 1.0 > 1.5 → False → no fire
            channels=[channel],
        )
        report = Runner().run(_plan(), _df(), alert_rules=[rule])
        assert report.fired_alerts == []
        channel.write.assert_not_called()

    def test_multiple_channels_all_notified(self):
        """All channels on a rule receive write() when it fires."""
        ch1, ch2 = MagicMock(), MagicMock()
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.5, upper_bound=True),
            channels=[ch1, ch2],
        )
        report = Runner().run(_plan(), _df(), alert_rules=[rule])
        ch1.write.assert_called_once_with(report)
        ch2.write.assert_called_once_with(report)

    def test_channel_exception_does_not_abort_run(self):
        """A channel that raises does not prevent the report from being returned."""
        bad_channel = MagicMock()
        bad_channel.write.side_effect = RuntimeError("SMTP down")
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.5, upper_bound=True),
            channels=[bad_channel],
        )
        report = Runner().run(_plan(), _df(), alert_rules=[rule])
        # The run completed and the alert is still recorded
        assert len(report.fired_alerts) == 1

    def test_channel_exception_logged_as_warning(self, caplog):
        bad_channel = MagicMock()
        bad_channel.write.side_effect = RuntimeError("timeout")
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.5, upper_bound=True),
            channels=[bad_channel],
        )
        with caplog.at_level(logging.WARNING, logger="ayn_ml.runner"):
            Runner().run(_plan(), _df(), alert_rules=[rule])
        assert any("channel.write() failed" in m for m in caplog.messages)

    def test_unknown_metric_name_logs_warning_and_no_alert(self, caplog):
        rule = AlertRule(
            metric_name="nonexistent",
            policy=ThresholdPolicy(),
            channels=[MagicMock()],
        )
        with caplog.at_level(logging.WARNING, logger="ayn_ml.runner"):
            report = Runner().run(_plan(), _df(), alert_rules=[rule])
        assert report.fired_alerts == []
        assert any("unknown metric" in m.lower() for m in caplog.messages)

    def test_status_based_policy_does_not_fire_when_metric_passes(self):
        """ThresholdPolicy() fires when MetricResult.status is False.

        MetricSpec(upper_bound=False, threshold=0.8) → passes when value >= 0.8.
        accuracy=1.0 >= 0.8 → status=True → ThresholdPolicy() does NOT fire.
        """
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(),  # delegates to MetricResult.status
            channels=[MagicMock()],
        )
        # upper_bound=False → passes when accuracy >= threshold → 1.0 >= 0.8 → status=True
        plan_pass = MonitoringPlan(
            name="test",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
            metrics=[MetricSpec(name="accuracy", threshold=0.8, upper_bound=False)],
        )
        report = Runner().run(plan_pass, _df(), alert_rules=[rule])
        assert report.fired_alerts == []

    def test_status_based_policy_fires_when_metric_fails(self):
        """ThresholdPolicy() fires when MetricResult.status is False.

        MetricSpec(upper_bound=True, threshold=0.5) → passes when value <= 0.5.
        accuracy=1.0 → status=False → ThresholdPolicy() fires.
        """
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(),
            channels=[],
        )
        # upper_bound=True → passes when accuracy <= threshold → 1.0 <= 0.5 → False → status=False
        plan_fail = MonitoringPlan(
            name="test",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
            metrics=[MetricSpec(name="accuracy", threshold=0.5, upper_bound=True)],
        )
        report = Runner().run(plan_fail, _df(), alert_rules=[rule])
        assert len(report.fired_alerts) == 1

    def test_fired_alert_details_contain_value_and_threshold(self):
        rule = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.5, upper_bound=True),
            channels=[],
        )
        report = Runner().run(_plan(), _df(), alert_rules=[rule])
        assert len(report.fired_alerts) == 1
        details = report.fired_alerts[0].details
        assert "value" in details
        assert "threshold" in details
        assert details["threshold"] == 0.5

    def test_two_rules_same_metric_both_channels_notified(self):
        """Regression: two AlertRules watching the same metric must both dispatch.

        Previously a dict comprehension in Step 12 silently dropped the first
        rule's channels when two rules shared the same metric_name.
        """
        ch1, ch2 = MagicMock(), MagicMock()
        rule1 = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.5, upper_bound=True),
            channels=[ch1],
        )
        rule2 = AlertRule(
            metric_name="accuracy",
            policy=ThresholdPolicy(threshold=0.3, upper_bound=True),
            channels=[ch2],
        )
        report = Runner().run(_plan(), _df(), alert_rules=[rule1, rule2])
        # Both rules fire (accuracy=1.0 > 0.5 and > 0.3)
        assert len(report.fired_alerts) == 2
        ch1.write.assert_called_once_with(report)
        ch2.write.assert_called_once_with(report)

    def test_one_rule_fires_per_matching_feature_column(self):
        """Regression: one AlertRule fires once per per-feature result.

        Previously the result index collapsed {name: result} so only the last
        PSI/Wasserstein result survived — the first was silently dropped.
        """
        import numpy as np

        rng = np.random.default_rng(0)
        n = 30
        ref = pd.DataFrame({
            "y_true": rng.integers(0, 2, n),
            "y_pred": rng.integers(0, 2, n),
            "age": rng.normal(40, 1, n),
            "income": rng.normal(1000, 10, n),
        })
        # Both features heavily shifted → Wasserstein >> 0 → rule fires for both
        cur = pd.DataFrame({
            "y_true": rng.integers(0, 2, n),
            "y_pred": rng.integers(0, 2, n),
            "age": rng.normal(60, 1, n),
            "income": rng.normal(2000, 10, n),
        })
        plan = MonitoringPlan(
            name="drift_test",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
            metrics=[
                MetricSpec(name="wasserstein", feature_name="age"),
                MetricSpec(name="wasserstein", feature_name="income"),
            ],
        )
        rule = AlertRule(
            metric_name="wasserstein",
            policy=ThresholdPolicy(threshold=0.0, upper_bound=True),  # fires when distance > 0
            channels=[],
        )
        report = Runner(strict=False).run(plan, cur, reference=ref, alert_rules=[rule])

        assert len(report.fired_alerts) == 2
        fired_features = {fa.feature_name for fa in report.fired_alerts}
        assert fired_features == {"age", "income"}
        for fa in report.fired_alerts:
            assert fa.metric_name == "wasserstein"
            assert fa.feature_name is not None
