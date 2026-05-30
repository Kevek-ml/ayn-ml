"""Tests for ayn_ml.advisor — MetricAdvisor and SuggestedPlan."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
import numpy as np

from ayn_ml.advisor import MetricAdvisor, SuggestedPlan
from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MonitoringPlan

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _schema(**kwargs) -> TabularSchema:
    return TabularSchema(label_col="y_true", prediction_col="y_pred", proba_col="y_prob", **kwargs)


def _balanced_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Binary classification, balanced classes, two numeric features."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(int)
    return pd.DataFrame(
        {
            "y_true": y,
            "y_pred": y,
            "y_prob": rng.uniform(0, 1, n),
            "age": rng.normal(35, 10, n),
            "income": rng.normal(50_000, 15_000, n),
        }
    )


def _ref_df(n: int = 200) -> pd.DataFrame:
    """Default reference for tests that don't exercise reference-specific routing."""
    return _balanced_df(n=n, seed=99)


def _imbalanced_df(ratio: float = 12.0, n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Binary classification with controlled imbalance ratio."""
    rng = np.random.default_rng(seed)
    n_minority = int(n / (1 + ratio))
    n_majority = n - n_minority
    y = np.concatenate([np.ones(n_majority, dtype=int), np.zeros(n_minority, dtype=int)])
    return pd.DataFrame(
        {
            "y_true": y,
            "y_pred": y,
            "y_prob": rng.uniform(0, 1, n),
            "age": rng.normal(35, 10, n),
        }
    )


def _regression_df(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "y_true": x + rng.normal(0, 0.1, n),
            "y_pred": x,
            "feature_a": rng.normal(0, 1, n),
        }
    )


def _categorical_df(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(int)
    cats = rng.choice(["A", "B", "C"], size=n)
    return pd.DataFrame(
        {
            "y_true": y,
            "y_pred": y,
            "y_prob": rng.uniform(0, 1, n),
            "region": cats,
        }
    )


def _small_df(n: int = 20, seed: int = 4) -> pd.DataFrame:
    """DataFrame with n < 30 rows."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(int)
    return pd.DataFrame(
        {
            "y_true": y,
            "y_pred": y,
            "y_prob": rng.uniform(0, 1, n),
            "score": rng.normal(0, 1, n),
        }
    )


def _large_df(n: int = 60_000, seed: int = 5) -> pd.DataFrame:
    """DataFrame with n > 50 000 rows."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(int)
    return pd.DataFrame(
        {
            "y_true": y,
            "y_pred": y,
            "y_prob": rng.uniform(0, 1, n),
            "score": rng.normal(0, 1, n),
        }
    )


# ---------------------------------------------------------------------------
# SuggestedPlan
# ---------------------------------------------------------------------------


class TestSuggestedPlan:
    def test_to_dict_has_plan_and_warnings(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        d = result.to_dict()
        assert "plan" in d
        assert "warnings" in d
        assert isinstance(d["warnings"], list)

    def test_to_dict_plan_is_serializable(self):
        import json

        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        # Should not raise
        json.dumps(result.to_dict(), default=str)

    def test_plan_is_monitoring_plan(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        assert isinstance(result.plan, MonitoringPlan)

    def test_warnings_is_tuple_of_strings(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        assert isinstance(result.warnings, tuple)
        assert all(isinstance(w, str) for w in result.warnings)

    def test_suggested_plan_is_frozen(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        with pytest.raises(Exception):
            result.warnings = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MetricAdvisor — basic API
# ---------------------------------------------------------------------------


class TestMetricAdvisorBasic:
    def test_returns_suggested_plan(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        assert isinstance(result, SuggestedPlan)

    def test_plan_name_propagated(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df(), name="my_plan")
        assert result.plan.name == "my_plan"

    def test_model_id_and_version_propagated(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df(), model_id="fraud_v2", model_version="1.0")
        assert result.plan.model_id == "fraud_v2"
        assert result.plan.model_version == "1.0"

    def test_invalid_task_type_raises_regression_variant(self):
        df = _balanced_df()
        with pytest.raises(ValueError, match="task_type"):
            MetricAdvisor(_schema()).suggest(df, reference=_ref_df(), task_type="unknown")

    def test_invalid_task_type_raises(self):
        df = _balanced_df()
        with pytest.raises(ValueError, match="task_type"):
            MetricAdvisor(_schema()).suggest(df, reference=_ref_df(), task_type="clustering")

    def test_reference_none_raises(self):
        df = _balanced_df()
        with pytest.raises((TypeError, ValueError)):
            MetricAdvisor(_schema()).suggest(df, reference=None)

    def test_reference_missing_raises(self):
        df = _balanced_df()
        with pytest.raises(TypeError):
            MetricAdvisor(_schema()).suggest(df)  # type: ignore[call-arg]

    def test_metrics_list_is_non_empty(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        assert len(result.plan.metrics) > 0

    def test_target_drift_always_included(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        names = [s.name for s in result.plan.metrics]
        assert "target_drift" in names


# ---------------------------------------------------------------------------
# Performance spec routing
# ---------------------------------------------------------------------------


class TestPerformanceSpecs:
    def _metric_names(self, df, reference=None, **kwargs) -> list[str]:
        if reference is None:
            reference = _ref_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=reference, **kwargs)
        return [s.name for s in result.plan.metrics]

    def test_balanced_classification_includes_accuracy(self):
        assert "accuracy" in self._metric_names(_balanced_df())

    def test_balanced_minimal_has_accuracy(self):
        assert "accuracy" in self._metric_names(_balanced_df())

    def test_imbalance_gt_10_excludes_accuracy_from_minimal(self):
        df = _imbalanced_df(ratio=12.0)
        names = self._metric_names(df)
        assert "accuracy" not in names

    def test_imbalance_gt_10_includes_f1_and_aucpr(self):
        df = _imbalanced_df(ratio=12.0)
        names = self._metric_names(df)
        assert "f1" in names
        assert "aucpr" in names

    def test_imbalance_gt_10_warning_emitted(self):
        df = _imbalanced_df(ratio=12.0)
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        assert any("accuracy excluded" in w for w in result.warnings)

    def test_imbalance_gt_5_demotes_accuracy(self):
        df = _imbalanced_df(ratio=7.0)
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        assert any("accuracy demoted" in w for w in result.warnings)

    def test_regression_minimal_has_mae_and_r2(self):
        df = _regression_df()
        names = self._metric_names(df, task_type="regression")
        assert "mae" in names
        assert "r2" in names

    def test_regression_minimal_no_accuracy(self):
        df = _regression_df()
        names = self._metric_names(df, task_type="regression")
        assert "accuracy" not in names


# ---------------------------------------------------------------------------
# Drift spec routing
# ---------------------------------------------------------------------------


class TestDriftSpecs:
    def _specs_for_col(self, df, col: str, reference=None, **kwargs) -> list[str]:
        if reference is None:
            reference = _ref_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=reference, **kwargs)
        return [s.name for s in result.plan.metrics if s.feature_name == col]

    def test_numeric_includes_psi_and_wasserstein(self):
        df = _balanced_df()
        names = self._specs_for_col(df, "age")
        assert "psi" in names
        assert "wasserstein" in names

    def test_numeric_normal_reference_uses_ttest(self):
        # Reference has a normal 'score' → normality routing on reference → ttest
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),
            }
        )
        ref = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),  # normal reference → ttest
            }
        )
        names = self._specs_for_col(df, "score", reference=ref)
        assert "ttest" in names or "mannwhitney" in names

    def test_normality_routing_uses_reference_distribution(self):
        """Normal reference → ttest; skewed reference → mannwhitney,
        regardless of the current window shape."""
        rng = np.random.default_rng(7)
        n = 300

        base = {
            "y_true": (np.arange(n) % 2).astype(int),
            "y_pred": (np.arange(n) % 2).astype(int),
            "y_prob": rng.uniform(0, 1, n),
        }

        # Current window is always skewed (exponential)
        df_cur = pd.DataFrame({**base, "score": rng.exponential(2.0, n)})

        # Normal reference → routing should produce ttest (or at least not mannwhitney)
        ref_normal = pd.DataFrame({**base, "score": rng.normal(0, 1, n)})
        names_normal_ref = self._specs_for_col(df_cur, "score", reference=ref_normal)

        # Skewed reference → routing should produce mannwhitney
        ref_skewed = pd.DataFrame({**base, "score": rng.exponential(2.0, n)})
        names_skewed_ref = self._specs_for_col(df_cur, "score", reference=ref_skewed)

        assert "mannwhitney" in names_skewed_ref
        assert "ttest" not in names_skewed_ref
        # Normal ref → parametric path
        assert "ttest" in names_normal_ref or "mannwhitney" in names_normal_ref

    def test_skewed_reference_column_uses_mannwhitney(self):
        rng = np.random.default_rng(42)
        n = 200
        # Reference has an exponential 'wait_time' — always skewed
        ref = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "wait_time": rng.exponential(scale=2.0, size=n),
            }
        )
        df = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "wait_time": rng.exponential(scale=2.0, size=n),
            }
        )
        names = self._specs_for_col(df, "wait_time", reference=ref)
        assert "mannwhitney" in names
        assert "ttest" not in names

    def test_categorical_gets_psi(self):
        df = _categorical_df()
        names = self._specs_for_col(df, "region")
        assert "psi" in names

    def test_categorical_gets_chisquare(self):
        df = _categorical_df()
        names = self._specs_for_col(df, "region")
        assert "chisquare" in names

    def test_categorical_no_ttest_or_mannwhitney(self):
        df = _categorical_df()
        names = self._specs_for_col(df, "region")
        assert "ttest" not in names
        assert "mannwhitney" not in names

    def test_small_n_wasserstein_only(self):
        df = _small_df()  # n=20 < 30
        names = self._specs_for_col(df, "score")
        assert "wasserstein" in names
        assert "ttest" not in names
        assert "mannwhitney" not in names
        assert "cramervonmises" not in names

    def test_small_n_warning_emitted(self):
        df = _small_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        assert any("too small" in w for w in result.warnings)

    def test_large_n_psi_and_wasserstein_only(self):
        df = _large_df()  # n=60_000 > 50_000
        names = self._specs_for_col(df, "score")
        assert "psi" in names
        assert "wasserstein" in names
        assert "ttest" not in names
        assert "mannwhitney" not in names

    def test_no_levene_when_reference_column_absent(self):
        """When a feature column is absent from the reference, variance ratio
        cannot be computed — Levene is not added for that column."""
        rng = np.random.default_rng(0)
        n = 200
        df = _balanced_df(n=n)  # has 'age' and 'income'
        # Reference has only 'age' — 'income' absent → no variance ratio → no Levene for income
        ref = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "age": rng.normal(35, 10, n),
            }
        )
        result = MetricAdvisor(_schema()).suggest(df, reference=ref)
        levene_for_income = [s for s in result.plan.metrics if s.name == "levene" and s.feature_name == "income"]
        assert levene_for_income == []

    def test_reference_with_high_variance_ratio_adds_levene(self):
        rng = np.random.default_rng(0)
        n = 200
        cur = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 5, n),  # std ≈ 5
            }
        )
        ref = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),  # std ≈ 1  → ratio ≈ 5 > 1.5
            }
        )
        result = MetricAdvisor(_schema()).suggest(cur, reference=ref)
        names = [s.name for s in result.plan.metrics]
        assert "levene" in names

    def test_reference_with_high_variance_ratio_warning_emitted(self):
        rng = np.random.default_rng(0)
        n = 200
        cur = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 5, n),
            }
        )
        ref = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),
            }
        )
        result = MetricAdvisor(_schema()).suggest(cur, reference=ref)
        assert any("levene added" in w for w in result.warnings)

    def test_ttest_has_equal_var_false_param(self):
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),
            }
        )
        ref = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),
            }
        )
        result = MetricAdvisor(_schema()).suggest(df, reference=ref)
        ttest_specs = [s for s in result.plan.metrics if s.name == "ttest"]
        for spec in ttest_specs:
            assert spec.params.get("equal_var") is False


# ---------------------------------------------------------------------------
# Statistics specs — minimal profile produces no descriptive stats
# ---------------------------------------------------------------------------


class TestStatisticsSpecs:
    def test_minimal_profile_no_stats(self):
        df = _balanced_df()
        result = MetricAdvisor(_schema()).suggest(df, reference=_ref_df())
        stat_names = {"mean", "std", "skewness", "kurtosis", "top_category"}
        names = {s.name for s in result.plan.metrics}
        assert names.isdisjoint(stat_names)


# ---------------------------------------------------------------------------
# Schema integration — feature_types declared
# ---------------------------------------------------------------------------


class TestSchemaIntegration:
    def test_declared_categorical_overrides_dtype_inference(self):
        rng = np.random.default_rng(0)
        n = 200
        # 'region' is an int column but declared categorical in schema
        df = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "region": rng.integers(0, 5, n),  # int dtype
            }
        )
        schema = _schema(feature_types={"region": "categorical"})
        result = MetricAdvisor(schema).suggest(df, reference=_ref_df())
        region_specs = [s.name for s in result.plan.metrics if s.feature_name == "region"]
        assert "psi" in region_specs
        assert "chisquare" in region_specs
        assert "ttest" not in region_specs

    def test_only_declared_features_analysed(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "age": rng.normal(35, 10, n),
                "undeclared": rng.normal(0, 1, n),
            }
        )
        schema = _schema(feature_types={"age": "numeric"})
        result = MetricAdvisor(schema).suggest(df, reference=_ref_df())
        # 'undeclared' should not appear in any spec's feature_name
        undeclared_specs = [s for s in result.plan.metrics if s.feature_name == "undeclared"]
        assert undeclared_specs == []


# ---------------------------------------------------------------------------
# Reuse across multiple calls
# ---------------------------------------------------------------------------


class TestDesignerReuse:
    def test_multiple_suggest_calls_independent(self):
        schema = _schema()
        designer = MetricAdvisor(schema)
        df1 = _balanced_df(n=200, seed=0)
        df2 = _imbalanced_df(ratio=12.0, seed=1)
        r1 = designer.suggest(df1, reference=_ref_df())
        r2 = designer.suggest(df2, reference=_ref_df())
        # r1 has accuracy (balanced), r2 does not (imbalanced)
        r1_names = [s.name for s in r1.plan.metrics]
        r2_names = [s.name for s in r2.plan.metrics]
        assert "accuracy" in r1_names
        assert "accuracy" not in r2_names

    def test_different_references_produce_different_routing(self):
        """Same current window, different references — routing can differ."""
        rng = np.random.default_rng(0)
        n = 300
        df_cur = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),
            }
        )
        ref_normal = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),  # normal → ttest routing
            }
        )
        ref_skewed = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.exponential(2.0, n),  # skewed → mannwhitney routing
            }
        )
        designer = MetricAdvisor(_schema())
        r_normal = designer.suggest(df_cur, reference=ref_normal)
        r_skewed = designer.suggest(df_cur, reference=ref_skewed)

        score_names_normal = [s.name for s in r_normal.plan.metrics if s.feature_name == "score"]
        score_names_skewed = [s.name for s in r_skewed.plan.metrics if s.feature_name == "score"]

        assert "mannwhitney" in score_names_skewed
        assert "ttest" not in score_names_skewed
        assert "ttest" in score_names_normal or "mannwhitney" in score_names_normal


# ---------------------------------------------------------------------------
# Polars backend — narwhals-compatibility smoke test
# ---------------------------------------------------------------------------


class TestPolarsBackend:
    def test_suggest_with_polars_dataframe(self):
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(0)
        n = 200
        df_pd = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "age": rng.normal(35, 10, n),
            }
        )
        ref_pd = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "age": rng.normal(35, 10, n),
            }
        )
        result = MetricAdvisor(_schema()).suggest(
            pl.from_pandas(df_pd), reference=pl.from_pandas(ref_pd)
        )
        assert isinstance(result, SuggestedPlan)
        assert len(result.plan.metrics) > 0

    def test_suggest_with_polars_reference(self):
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(0)
        n = 200
        cur_pd = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 5, n),
            }
        )
        ref_pd = pd.DataFrame(
            {
                "y_true": (np.arange(n) % 2).astype(int),
                "y_pred": (np.arange(n) % 2).astype(int),
                "y_prob": rng.uniform(0, 1, n),
                "score": rng.normal(0, 1, n),
            }
        )
        result = MetricAdvisor(_schema()).suggest(pl.from_pandas(cur_pd), reference=pl.from_pandas(ref_pd))
        assert isinstance(result, SuggestedPlan)
        names = [s.name for s in result.plan.metrics]
        assert "levene" in names
