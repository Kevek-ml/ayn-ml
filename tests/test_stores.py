"""Tests for ayn_ml.stores — InMemoryStore and SqliteStore."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import ayn_ml.metrics  # noqa: F401 — registers built-in metrics
from ayn_ml.core.result import (
    ExecutionContext,
    FiredAlert,
    MetricError,
    MetricResult,
    MonitoringReport,
)
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MonitoringPlan
from ayn_ml.stores import InMemoryStore, SqliteStore

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _plan(name: str = "test_plan", model_id: str = "m", model_version: str = "1") -> MonitoringPlan:
    return MonitoringPlan(
        name=name,
        model_id=model_id,
        model_version=model_version,
        data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
        metrics=[
            MetricSpec(name="accuracy"),
            MetricSpec(name="mean", feature_name="age"),
        ],
    )


def _ctx(
    run_id: str = "run1",
    model_id: str = "m",
    model_version: str = "1",
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id,
        model_id=model_id,
        model_version=model_version,
        eval_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        period_start=period_start or datetime(2026, 1, 1, tzinfo=timezone.utc),
        period_end=period_end or datetime(2026, 1, 7, tzinfo=timezone.utc),
        n_current=100,
        n_reference=500,
    )


def _result(name: str = "accuracy", value: float = 0.9, feature: str | None = None) -> MetricResult:
    return MetricResult(
        spec=MetricSpec(name=name, feature_name=feature),
        value=value,
        status=True,
    )


def _report(
    run_id: str = "run1",
    model_id: str = "m",
    model_version: str = "1",
    results: list[MetricResult] | None = None,
    errors: list[MetricError] | None = None,
    fired_alerts: list[FiredAlert] | None = None,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    profile: dict | None = None,
) -> MonitoringReport:
    return MonitoringReport(
        plan=_plan(model_id=model_id, model_version=model_version),
        context=_ctx(
            run_id=run_id,
            model_id=model_id,
            model_version=model_version,
            period_start=period_start,
            period_end=period_end,
        ),
        results=results or [_result()],
        errors=errors or [],
        fired_alerts=fired_alerts or [],
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Shared contract tests — run against both stores
# ---------------------------------------------------------------------------


class _StoreContractTests:
    """Mixin: tests that must pass for any ResultStore implementation."""

    def make_store(self):
        raise NotImplementedError

    def test_write_and_get_report(self):
        store = self.make_store()
        r = _report(run_id="abc")
        store.write(r)
        retrieved = store.get_report("abc")
        assert retrieved is not None
        assert retrieved.context.run_id == "abc"
        assert len(retrieved.results) == 1

    def test_get_report_unknown_run_returns_none(self):
        store = self.make_store()
        assert store.get_report("nonexistent") is None

    def test_read_history_returns_flat_dicts(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="fraud"))
        rows = store.read_history("fraud")
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert "metric_name" in rows[0]
        assert "value" in rows[0]
        assert "period_start" in rows[0]

    def test_read_history_filters_by_model_id(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="fraud"))
        store.write(_report(run_id="r2", model_id="churn"))
        rows = store.read_history("fraud")
        assert all(r["model_id"] == "fraud" for r in rows)

    def test_read_history_filters_by_model_version(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="m", model_version="v1"))
        store.write(_report(run_id="r2", model_id="m", model_version="v2"))
        rows = store.read_history("m", model_version="v1")
        assert all(r["model_version"] == "v1" for r in rows)
        assert len(rows) == 1

    def test_read_history_filters_by_metric_name(self):
        store = self.make_store()
        results = [_result("accuracy", 0.9), _result("f1", 0.85)]
        store.write(_report(run_id="r1", model_id="m", results=results))
        rows = store.read_history("m", metric_name="accuracy")
        assert len(rows) == 1
        assert rows[0]["metric_name"] == "accuracy"

    def test_read_history_limit(self):
        store = self.make_store()
        for i in range(5):
            store.write(_report(run_id=f"r{i}", model_id="m"))
        rows = store.read_history("m", limit=2)
        assert len(rows) == 2

    def test_read_history_no_limit_returns_all(self):
        store = self.make_store()
        for i in range(5):
            store.write(_report(run_id=f"r{i}", model_id="m"))
        rows = store.read_history("m")
        assert len(rows) == 5

    def test_read_history_get_metadata_adds_plan_fields(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="m"))
        rows = store.read_history("m", get_metadata=True)
        assert len(rows) == 1
        row = rows[0]
        assert "plan_name" in row
        assert "plan_window_type" in row
        assert "plan_sampling_type" in row
        assert "run_n_current" in row
        assert "run_n_reference" in row

    def test_read_history_no_metadata_omits_plan_fields(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="m"))
        rows = store.read_history("m", get_metadata=False)
        assert "plan_name" not in rows[0]
        assert "run_n_current" not in rows[0]

    def test_read_history_empty_when_no_match(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="fraud"))
        assert store.read_history("nonexistent") == []

    def test_write_report_with_errors_and_alerts(self):
        store = self.make_store()
        errors = [MetricError(metric_name="auc", error_type="SchemaError", message="oops")]
        alerts = [FiredAlert(metric_name="auc", policy_type="threshold", details={"value": 0.5})]
        r = _report(run_id="r1", model_id="m", errors=errors, fired_alerts=alerts)
        store.write(r)
        retrieved = store.get_report("r1")
        assert len(retrieved.errors) == 1
        assert retrieved.errors[0].metric_name == "auc"
        assert len(retrieved.fired_alerts) == 1
        assert retrieved.fired_alerts[0].policy_type == "threshold"

    def test_roundtrip_metric_value(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="m", results=[_result("auc", 0.876)]))
        rows = store.read_history("m", metric_name="auc")
        assert abs(rows[0]["value"] - 0.876) < 1e-6

    def test_get_report_preserves_period_dates(self):
        store = self.make_store()
        period_start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        period_end = datetime(2026, 3, 8, tzinfo=timezone.utc)
        store.write(
            _report(
                run_id="r1",
                model_id="m",
                period_start=period_start,
                period_end=period_end,
            )
        )
        retrieved = store.get_report("r1")
        assert retrieved is not None
        assert retrieved.context.period_start == period_start
        assert retrieved.context.period_end == period_end

    def test_get_report_preserves_profile(self):
        store = self.make_store()
        profile = {
            "age": {"mean": 40.0, "std": 10.0, "null_rate": 0.0},
            "income": {"mean": 50000.0, "null_rate": 0.05},
        }
        store.write(_report(run_id="r1", model_id="m", profile=profile))
        retrieved = store.get_report("r1")
        assert retrieved is not None
        assert retrieved.profile == profile

    def test_read_history_includes_profile_rows(self):
        store = self.make_store()
        profile = {"age": {"mean": 40.0, "std": 10.0}}
        store.write(_report(run_id="r1", model_id="m", profile=profile))
        rows = store.read_history("m")
        types = {r["metric_type"] for r in rows}
        assert "profile" in types
        profile_rows = [r for r in rows if r["metric_type"] == "profile"]
        assert len(profile_rows) == 2  # mean + std
        assert all(r["feature_name"] == "age" for r in profile_rows)

    def test_read_history_metric_type_filter(self):
        store = self.make_store()
        profile = {"age": {"mean": 40.0}}
        store.write(_report(run_id="r1", model_id="m", profile=profile))
        profile_rows = store.read_history("m", metric_type="profile")
        assert len(profile_rows) == 1
        assert profile_rows[0]["metric_type"] == "profile"
        assert profile_rows[0]["metric_name"] == "mean"
        assert profile_rows[0]["feature_name"] == "age"

    def test_read_history_combined_metric_name_and_metric_type_filter(self):
        # Report has: MetricResult "accuracy" + profile stat "mean" on "age"
        # Both metric_name="mean" and metric_type="profile" must agree for a row to match.
        store = self.make_store()
        profile = {"age": {"mean": 40.0, "std": 10.0}}
        store.write(
            _report(
                run_id="r1",
                model_id="m",
                results=[_result("mean", 0.9, feature="age")],  # MetricResult named "mean"
                profile=profile,
            )
        )
        # metric_type="profile" + metric_name="mean" → only the profile row, not the MetricResult
        rows = store.read_history("m", metric_name="mean", metric_type="profile")
        assert len(rows) == 1
        assert rows[0]["metric_type"] == "profile"
        assert rows[0]["feature_name"] == "age"
        # metric_name="std" + metric_type="profile" → one profile row
        rows_std = store.read_history("m", metric_name="std", metric_type="profile")
        assert len(rows_std) == 1
        assert rows_std[0]["metric_name"] == "std"

    def test_dataframe_compatible(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="m"))
        rows = store.read_history("m")
        df = pd.DataFrame(rows)
        assert "metric_name" in df.columns
        assert "value" in df.columns

    def test_write_is_thread_safe(self):
        store = self.make_store()
        errors = []

        def _write(i: int) -> None:
            try:
                store.write(_report(run_id=f"r{i}", model_id="m"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(store.read_history("m")) == 10


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class TestInMemoryStore(_StoreContractTests):
    def make_store(self):
        return InMemoryStore()

    def test_maxlen_evicts_oldest(self):
        store = InMemoryStore(maxlen=3)
        for i in range(5):
            store.write(_report(run_id=f"r{i}", model_id="m"))
        # Only last 3 reports retained
        assert store.get_report("r0") is None
        assert store.get_report("r1") is None
        assert store.get_report("r4") is not None

    def test_maxlen_none_unbounded(self):
        store = InMemoryStore(maxlen=None)
        for i in range(20):
            store.write(_report(run_id=f"r{i}", model_id="m"))
        assert len(store.read_history("m")) == 20


# ---------------------------------------------------------------------------
# SqliteStore
# ---------------------------------------------------------------------------


class TestSqliteStore(_StoreContractTests):
    def make_store(self):
        return SqliteStore(":memory:")

    def test_context_manager(self):
        with SqliteStore(":memory:") as store:
            store.write(_report(run_id="r1", model_id="m"))
            rows = store.read_history("m")
        assert len(rows) == 1

    def test_duplicate_run_id_ignored(self):
        store = self.make_store()
        r = _report(run_id="dup", model_id="m")
        store.write(r)
        store.write(r)  # second write with same run_id — should not raise
        assert len(store.read_history("m")) == 1

    def test_persistent_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        store1 = SqliteStore(db_path)
        store1.write(_report(run_id="r1", model_id="m"))
        store1.close()

        store2 = SqliteStore(db_path)
        rows = store2.read_history("m")
        store2.close()
        assert len(rows) == 1
        assert rows[0]["run_id"] == "r1"

    def test_get_report_reconstructs_plan(self):
        store = self.make_store()
        store.write(_report(run_id="r1", model_id="m"))
        retrieved = store.get_report("r1")
        assert retrieved.plan.name == "test_plan"
        assert retrieved.plan.data_schema.label_col == "y_true"

    def test_read_history_ordered_newest_first(self):
        store = self.make_store()
        dates = [
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 15, tzinfo=timezone.utc),
            datetime(2026, 1, 8, tzinfo=timezone.utc),
        ]
        for i, d in enumerate(dates):
            store.write(_report(run_id=f"r{i}", model_id="m", period_start=d))
        rows = store.read_history("m")
        periods = [r["period_start"] for r in rows]
        assert periods == sorted(periods, reverse=True)


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


class TestRunnerWithStore:
    def _df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n = 50
        return pd.DataFrame(
            {
                "y_true": rng.integers(0, 2, n),
                "y_pred": rng.integers(0, 2, n),
                "age": rng.normal(40, 10, n),
            }
        )

    def test_runner_writes_to_in_memory_store(self):
        from ayn_ml.runner import Runner

        store = InMemoryStore()
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
            metrics=[MetricSpec(name="accuracy")],
        )
        report = Runner(strict=False).run(plan, self._df(), store=store)
        rows = store.read_history("m", metric_name="accuracy")
        assert len(rows) == 1
        assert rows[0]["run_id"] == report.context.run_id

    def test_runner_writes_to_sqlite_store(self):
        from ayn_ml.runner import Runner

        store = SqliteStore(":memory:")
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
            metrics=[MetricSpec(name="accuracy")],
        )
        report = Runner(strict=False).run(plan, self._df(), store=store)
        rows = store.read_history("m", metric_name="accuracy")
        assert len(rows) == 1
        assert abs(rows[0]["value"] - report.results[0].value) < 1e-9

    def test_store_failure_does_not_abort_run(self):
        from ayn_ml.runner import Runner

        bad_store = InMemoryStore()
        bad_store.write = lambda r: (_ for _ in ()).throw(RuntimeError("disk full"))
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
            metrics=[MetricSpec(name="accuracy")],
        )
        report = Runner(strict=False).run(plan, self._df(), store=bad_store)
        assert report is not None
        assert len(report.results) == 1
