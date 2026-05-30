import numpy as np
import pandas as pd
import pytest

from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec
from ayn_ml.exceptions import InsufficientDataError, MetricComputeError, SchemaError
from ayn_ml.metrics import get_metric


def spec(name: str, **kwargs) -> MetricSpec:
    return MetricSpec(name=name, **kwargs)


def drift_spec(name: str, feature: str = "age", **kwargs) -> MetricSpec:
    return MetricSpec(name=name, feature_name=feature, **kwargs)


def stat_spec(name: str, feature: str = "age", **kwargs) -> MetricSpec:
    return MetricSpec(name=name, feature_name=feature, **kwargs)


class TestComputeStatusAndContext:
    def test_status_none_when_no_threshold(self, df_current, tabular_schema):
        r = get_metric("accuracy").compute(df_current, None, tabular_schema, spec("accuracy"))
        assert r.status is None

    def test_status_true_when_above_threshold(self, df_current, tabular_schema):
        r = get_metric("accuracy").compute(
            df_current,
            None,
            tabular_schema,
            spec("accuracy", threshold=0.0, upper_bound=False),
        )
        assert r.status is True

    def test_status_false_when_below_threshold(self, df_current, tabular_schema):
        r = get_metric("accuracy").compute(
            df_current,
            None,
            tabular_schema,
            spec("accuracy", threshold=1.0, upper_bound=False),
        )
        assert r.status is False

    def test_status_true_interval_threshold_inside(self, df_current, tabular_schema):
        r = get_metric("accuracy").compute(
            df_current,
            None,
            tabular_schema,
            spec("accuracy", threshold=[0.0, 1.0]),
        )
        assert r.status is True

    def test_status_false_interval_threshold_outside(self, df_current, tabular_schema):
        r = get_metric("accuracy").compute(
            df_current,
            None,
            tabular_schema,
            spec("accuracy", threshold=[0.99, 1.0]),
        )
        assert r.status is False


class TestPerformanceMetrics:
    @pytest.mark.parametrize("name", ["accuracy", "precision", "recall", "f1"])
    def test_classification_metrics_in_range(self, df_current, tabular_schema, name):
        r = get_metric(name).compute(df_current, None, tabular_schema, spec(name))
        assert 0.0 <= r.value <= 1.0

    @pytest.mark.parametrize("name", ["mse", "mae"])
    def test_regression_metrics_non_negative(self, df_current, tabular_schema, name):
        r = get_metric(name).compute(df_current, None, tabular_schema, spec(name))
        assert r.value >= 0.0

    def test_mape_positive(self, df_current, tabular_schema):
        df = df_current.copy()
        df["y_true"] = np.abs(df_current["y_true"].astype(float)) + 1.0
        r = get_metric("mape").compute(df, None, tabular_schema, spec("mape"))
        assert r.value >= 0.0

    def test_mape_raises_when_all_zeros(self, df_current, tabular_schema):
        df = df_current.copy()
        df["y_true"] = 0.0
        with pytest.raises(InsufficientDataError, match="zero"):
            get_metric("mape").compute(df, None, tabular_schema, spec("mape"))

    def test_aucpr_in_range(self, df_current, tabular_schema):
        r = get_metric("aucpr").compute(df_current, None, tabular_schema, spec("aucpr"))
        assert 0.0 <= r.value <= 1.0

    def test_brier_in_range(self, df_current, tabular_schema):
        r = get_metric("brier").compute(df_current, None, tabular_schema, spec("brier"))
        assert 0.0 <= r.value <= 1.0

    def test_auc_requires_proba(self, df_current):
        with pytest.raises(SchemaError, match="Probability column"):
            s = TabularSchema(proba_col=None)
            get_metric("auc").compute(df_current, None, s, spec("auc"))

    def test_log_loss_positive(self, df_current, tabular_schema):
        r = get_metric("log_loss").compute(df_current, None, tabular_schema, spec("log_loss"))
        assert r.value > 0.0

    def test_r2_returns_float(self, df_current, tabular_schema):
        r = get_metric("r2").compute(df_current, None, tabular_schema, spec("r2"))
        assert isinstance(r.value, float)

    def test_r2_raises_on_constant_y_true(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        sklearn = pytest.importorskip("sklearn")  # noqa: F841
        df = pd.DataFrame({"y_true": [1.0] * 20, "y_pred": [1.0] * 20})
        with pytest.raises(MetricComputeError, match="identical"):
            get_metric("r2").compute(df, None, tabular_schema, spec("r2"))

    def test_auc_raises_on_single_class(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        sklearn = pytest.importorskip("sklearn")  # noqa: F841
        df = pd.DataFrame({"y_true": [0] * 20, "y_pred": [0] * 20, "y_pred_proba": [0.3] * 20})
        with pytest.raises(MetricComputeError):
            get_metric("auc").compute(df, None, tabular_schema, spec("auc"))

    def test_aucpr_raises_on_single_class(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        sklearn = pytest.importorskip("sklearn")  # noqa: F841
        df = pd.DataFrame({"y_true": [1] * 20, "y_pred": [1] * 20, "y_pred_proba": [0.8] * 20})
        with pytest.raises(MetricComputeError):
            get_metric("aucpr").compute(df, None, tabular_schema, spec("aucpr"))

    def test_missing_label_col_raises(self, df_current):
        bad_schema = TabularSchema(label_col="missing_col")
        with pytest.raises(SchemaError, match="missing_col"):
            get_metric("accuracy").compute(df_current, None, bad_schema, spec("accuracy"))

    def test_insufficient_data_raises(self, df_current, tabular_schema):
        tiny = df_current.iloc[:1]
        with pytest.raises(InsufficientDataError):
            get_metric("accuracy").compute(tiny, None, tabular_schema, spec("accuracy"))


class TestDriftMetrics:
    def test_psi_positive(self, df_current, df_reference, tabular_schema):
        r = get_metric("psi").compute(df_current, df_reference, tabular_schema, drift_spec("psi"))
        assert r.value >= 0.0

    def test_psi_low_when_no_drift(self, df_reference, tabular_schema):
        r = get_metric("psi").compute(df_reference, df_reference, tabular_schema, drift_spec("psi"))
        assert r.value < 0.05

    def test_psi_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError, match="requires reference"):
            get_metric("psi").compute(df_current, None, tabular_schema, drift_spec("psi"))

    def test_wasserstein_positive(self, df_current, df_reference, tabular_schema):
        r = get_metric("wasserstein").compute(df_current, df_reference, tabular_schema, drift_spec("wasserstein"))
        assert r.value >= 0.0

    def test_mmd_non_negative(self, df_current, df_reference, tabular_schema):
        r = get_metric("mmd").compute(df_current, df_reference, tabular_schema, drift_spec("mmd"))
        assert r.value >= 0.0

    def test_mmd_subsampling_path(self, df_current, df_reference, tabular_schema):
        s = MetricSpec(
            name="mmd",
            feature_name="age",
            params={"max_samples": 5, "random_state": 42},
        )
        r = get_metric("mmd").compute(df_current, df_reference, tabular_schema, s)
        assert r.value >= 0.0

    def test_psi_categorical_non_negative(self, df_current, df_reference, tabular_schema):
        r = get_metric("psi").compute(df_current, df_reference, tabular_schema, drift_spec("psi", feature="category"))
        assert r.value >= 0.0

    def test_psi_categorical_zero_when_no_drift(self, df_reference, tabular_schema):
        r = get_metric("psi").compute(df_reference, df_reference, tabular_schema, drift_spec("psi", feature="category"))
        assert r.value < 0.05

    def test_wasserstein_raises_on_categorical(self, df_current, df_reference, tabular_schema):
        with pytest.raises(SchemaError, match="numeric"):
            get_metric("wasserstein").compute(
                df_current, df_reference, tabular_schema, drift_spec("wasserstein", feature="category")
            )

    def test_mmd_raises_on_categorical(self, df_current, df_reference, tabular_schema):
        with pytest.raises(SchemaError, match="numeric"):
            get_metric("mmd").compute(df_current, df_reference, tabular_schema, drift_spec("mmd", feature="category"))

    def test_psi_warns_and_nonzero_on_out_of_range_current(self, df_reference, tabular_schema, caplog):
        pd = pytest.importorskip("pandas")
        import logging

        ref = df_reference
        # half of current is far beyond the reference range
        cur = pd.DataFrame({"age": [ref["age"].max() + 50.0] * 50 + list(ref["age"][:50])})
        with caplog.at_level(logging.WARNING, logger="ayn_ml.metrics.tabular.drift"):
            r = get_metric("psi").compute(cur, ref, tabular_schema, drift_spec("psi", feature="age"))
        assert r.value > 0.1
        assert any("outside the reference range" in m for m in caplog.messages)

    def test_psi_int_encoded_categorical_via_feature_types(self, df_current, df_reference):
        schema = TabularSchema(feature_types={"age": "categorical"})
        r = get_metric("psi").compute(df_current, df_reference, schema, drift_spec("psi", feature="age"))
        assert r.value >= 0.0

    def test_requires_feature_name(self, df_current, df_reference, tabular_schema):
        bad_spec = MetricSpec(name="psi")
        with pytest.raises(SchemaError, match="feature_name"):
            get_metric("psi").compute(df_current, df_reference, tabular_schema, bad_spec)


class TestStatisticalTests:
    @pytest.mark.parametrize("name", ["ks_2samp", "ttest", "mannwhitney", "levene", "cramervonmises"])
    def test_pvalue_in_range(self, df_current, df_reference, tabular_schema, name):
        r = get_metric(name).compute(df_current, df_reference, tabular_schema, drift_spec(name))
        assert 0.0 <= r.value <= 1.0

    def test_ks_detects_drift(self, df_current, df_reference, tabular_schema):
        r = get_metric("ks_2samp").compute(df_current, df_reference, tabular_schema, drift_spec("ks_2samp"))
        assert r.value < 0.05

    def test_ttest_welch_variant(self, df_current, df_reference, tabular_schema):
        r = get_metric("ttest").compute(
            df_current, df_reference, tabular_schema, drift_spec("ttest", params={"equal_var": False})
        )
        assert 0.0 <= r.value <= 1.0

    def test_mannwhitney_one_sided(self, df_current, df_reference, tabular_schema):
        r = get_metric("mannwhitney").compute(
            df_current, df_reference, tabular_schema, drift_spec("mannwhitney", params={"alternative": "greater"})
        )
        assert 0.0 <= r.value <= 1.0

    @pytest.mark.parametrize("name", ["ks_2samp", "ttest", "mannwhitney", "levene", "cramervonmises"])
    def test_requires_reference(self, df_current, tabular_schema, name):
        with pytest.raises(SchemaError, match="requires reference"):
            get_metric(name).compute(df_current, None, tabular_schema, drift_spec(name))

    @pytest.mark.parametrize("name", ["ks_2samp", "ttest", "mannwhitney", "levene", "cramervonmises"])
    def test_missing_column_raises(self, df_current, df_reference, tabular_schema, name):
        with pytest.raises(SchemaError, match="not found"):
            get_metric(name).compute(df_current, df_reference, tabular_schema, drift_spec(name, feature="nonexistent"))

    @pytest.mark.parametrize("name", ["ks_2samp", "ttest", "mannwhitney", "levene"])
    def test_effect_size_is_float(self, df_current, df_reference, tabular_schema, name):
        r = get_metric(name).compute(df_current, df_reference, tabular_schema, drift_spec(name))
        assert isinstance(r.effect_size, float)
        assert r.effect_size_label is not None

    def test_ttest_effect_size_cohens_d_formula(self, df_current, df_reference, tabular_schema):
        r = get_metric("ttest").compute(df_current, df_reference, tabular_schema, drift_spec("ttest"))
        assert r.effect_size is not None
        assert abs(r.effect_size) > 0
        assert r.effect_size_label == "cohen_d"

    def test_mannwhitney_cliffs_delta_in_range(self, df_current, df_reference, tabular_schema):
        r = get_metric("mannwhitney").compute(df_current, df_reference, tabular_schema, drift_spec("mannwhitney"))
        assert -1.0 <= r.effect_size <= 1.0
        assert r.effect_size_label == "cliff_delta"

    def test_levene_variance_ratio_positive(self, df_current, df_reference, tabular_schema):
        r = get_metric("levene").compute(df_current, df_reference, tabular_schema, drift_spec("levene"))
        assert r.effect_size > 0.0
        assert r.effect_size_label == "variance_ratio"

    def test_levene_zero_variance_reference_returns_none_effect_size(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        ref = pd.DataFrame({"age": [5.0] * 20})  # constant → var = 0
        cur = pd.DataFrame({"age": [5.0, 6.0] * 10})
        r = get_metric("levene").compute(cur, ref, tabular_schema, drift_spec("levene"))
        assert r.effect_size is None
        assert r.effect_size_label is None

    def test_ks_2samp_effect_size_label(self, df_current, df_reference, tabular_schema):
        r = get_metric("ks_2samp").compute(df_current, df_reference, tabular_schema, drift_spec("ks_2samp"))
        assert r.effect_size_label == "ks_statistic"

    def test_cramervonmises_no_effect_size(self, df_current, df_reference, tabular_schema):
        r = get_metric("cramervonmises").compute(df_current, df_reference, tabular_schema, drift_spec("cramervonmises"))
        assert r.effect_size is None
        assert r.effect_size_label is None

    def test_ttest_student_variant_explicit(self, df_current, df_reference, tabular_schema):
        r = get_metric("ttest").compute(
            df_current, df_reference, tabular_schema, drift_spec("ttest", params={"equal_var": True})
        )
        assert 0.0 <= r.value <= 1.0


# ── Chi-square ─────────────────────────────────────────────────────────────────


class TestChiSquare:
    def test_pvalue_in_range(self, df_current, df_reference, tabular_schema):
        r = get_metric("chisquare").compute(
            df_current, df_reference, tabular_schema, drift_spec("chisquare", feature="category")
        )
        assert 0.0 <= r.value <= 1.0

    def test_no_drift_high_pvalue(self, df_reference, tabular_schema):
        r = get_metric("chisquare").compute(
            df_reference, df_reference, tabular_schema, drift_spec("chisquare", feature="category")
        )
        assert r.value > 0.05

    def test_large_drift_low_pvalue(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        ref = pd.DataFrame({"cat": ["A"] * 150 + ["B"] * 150})
        cur = pd.DataFrame({"cat": ["A"] * 10 + ["B"] * 290})
        r = get_metric("chisquare").compute(cur, ref, tabular_schema, drift_spec("chisquare", feature="cat"))
        assert r.value < 0.05

    def test_cramer_v_in_range(self, df_current, df_reference, tabular_schema):
        r = get_metric("chisquare").compute(
            df_current, df_reference, tabular_schema, drift_spec("chisquare", feature="category")
        )
        assert 0.0 <= r.effect_size <= 1.0
        assert r.effect_size_label == "cramer_v"

    def test_cramer_v_zero_when_no_drift(self, df_reference, tabular_schema):
        r = get_metric("chisquare").compute(
            df_reference, df_reference, tabular_schema, drift_spec("chisquare", feature="category")
        )
        assert r.effect_size == pytest.approx(0.0, abs=1e-6)

    def test_integer_labels_accepted(self, df_current, df_reference, tabular_schema):
        r = get_metric("chisquare").compute(
            df_current, df_reference, tabular_schema, drift_spec("chisquare", feature="y_true")
        )
        assert 0.0 <= r.value <= 1.0

    def test_float_column_raises(self, df_current, df_reference, tabular_schema):
        with pytest.raises(SchemaError, match="categorical"):
            get_metric("chisquare").compute(
                df_current, df_reference, tabular_schema, drift_spec("chisquare", feature="age")
            )

    def test_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError, match="requires reference"):
            get_metric("chisquare").compute(
                df_current, None, tabular_schema, drift_spec("chisquare", feature="category")
            )

    def test_missing_column_raises(self, df_current, df_reference, tabular_schema):
        with pytest.raises(SchemaError, match="not found"):
            get_metric("chisquare").compute(
                df_current, df_reference, tabular_schema, drift_spec("chisquare", feature="nonexistent")
            )

    def test_single_category_returns_p1(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        ref = pd.DataFrame({"cat": ["A"] * 50})
        cur = pd.DataFrame({"cat": ["A"] * 50})
        r = get_metric("chisquare").compute(cur, ref, tabular_schema, drift_spec("chisquare", feature="cat"))
        assert r.value == pytest.approx(1.0)
        assert r.effect_size == pytest.approx(0.0)

    def test_new_category_in_current(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        ref = pd.DataFrame({"cat": ["A"] * 150 + ["B"] * 150})
        cur = pd.DataFrame({"cat": ["A"] * 100 + ["B"] * 100 + ["C"] * 100})
        r = get_metric("chisquare").compute(cur, ref, tabular_schema, drift_spec("chisquare", feature="cat"))
        assert r.value < 0.05

    def test_threshold_status_pass(self, df_reference, tabular_schema):
        # Same distribution → p ≈ 1.0 ≥ 0.05 → status True (no drift)
        r = get_metric("chisquare").compute(
            df_reference,
            df_reference,
            tabular_schema,
            drift_spec("chisquare", feature="category", threshold=0.05, upper_bound=False),
        )
        assert r.status is True

    def test_threshold_status_fail(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        # Large drift → p ≪ 0.05 → status False (drift detected)
        ref = pd.DataFrame({"cat": ["A"] * 150 + ["B"] * 150})
        cur = pd.DataFrame({"cat": ["A"] * 10 + ["B"] * 290})
        r = get_metric("chisquare").compute(
            cur,
            ref,
            tabular_schema,
            drift_spec("chisquare", feature="cat", threshold=0.05, upper_bound=False),
        )
        assert r.status is False

    def test_float_column_on_current_raises(self, df_reference, tabular_schema):
        pd = pytest.importorskip("pandas")
        cur_float = pd.DataFrame({"age": [1.5, 2.5] * 150})
        ref_int = pd.DataFrame({"age": [1, 2] * 150})
        with pytest.raises(SchemaError, match="categorical"):
            get_metric("chisquare").compute(cur_float, ref_int, tabular_schema, drift_spec("chisquare", feature="age"))


class TestStatisticsMetrics:
    def test_mean_close_to_expected(self, df_current, tabular_schema):
        r = get_metric("mean").compute(df_current, None, tabular_schema, stat_spec("mean"))
        assert 45.0 < r.value < 55.0

    def test_median_returns_float(self, df_current, tabular_schema):
        r = get_metric("median").compute(df_current, None, tabular_schema, stat_spec("median"))
        assert isinstance(r.value, float)

    def test_std_positive(self, df_current, tabular_schema):
        r = get_metric("std").compute(df_current, None, tabular_schema, stat_spec("std"))
        assert r.value > 0.0

    def test_count_equals_n(self, df_current, tabular_schema):
        r = get_metric("count").compute(df_current, None, tabular_schema, stat_spec("count"))
        assert r.value == 300

    def test_count_returns_zero_on_empty_window(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        empty = pd.DataFrame({"age": []})
        r = get_metric("count").compute(empty, None, tabular_schema, stat_spec("count"))
        assert r.value == 0

    def test_quantile_p95(self, df_current, tabular_schema):
        p95 = get_metric("quantile").compute(
            df_current, None, tabular_schema, stat_spec("quantile", params={"q": 0.95})
        )
        p50 = get_metric("median").compute(df_current, None, tabular_schema, stat_spec("median"))
        assert p95.value > p50.value

    def test_top_category_returns_string(self, df_current, tabular_schema):
        r = get_metric("top_category").compute(
            df_current, None, tabular_schema, stat_spec("top_category", feature="category")
        )
        assert isinstance(r.value, str)

    def test_skewness_and_kurtosis_finite(self, df_current, tabular_schema):
        for name in ("skewness", "kurtosis"):
            r = get_metric(name).compute(df_current, None, tabular_schema, stat_spec(name))
            assert np.isfinite(r.value)


class TestColumnTypeRouting:
    """Runner routes metrics by column kind; incompatible columns produce SchemaError in the report."""

    def _run(self, metric_name: str, feature_name: str, df, schema=None):
        from ayn_ml.core.schema import TabularSchema
        from ayn_ml.core.spec import MonitoringPlan
        from ayn_ml.runner import Runner

        plan = MonitoringPlan(
            name="t",
            model_id="m",
            model_version="1",
            data_schema=schema or TabularSchema(),
            metrics=[MetricSpec(name=metric_name, feature_name=feature_name)],
        )
        return Runner(strict=False).run(plan, df, reference=df)

    def _df(self, n: int = 50) -> "pd.DataFrame":
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "y_true": rng.integers(0, 2, n),
                "y_pred": rng.integers(0, 2, n),
                "y_pred_proba": rng.uniform(0, 1, n),
                "age": rng.normal(40, 10, n),
                "age_str": [f"val_{i % 5}" for i in range(n)],
                "flag": rng.integers(0, 2, n),
            }
        )

    def _assert_schema_error(self, report, metric_name: str, fragment: str):
        assert len(report.errors) == 1
        err = report.errors[0]
        assert err.metric_name == metric_name
        assert err.error_type == "SchemaError"
        assert fragment in err.message

    def test_categorical_column_to_ks_2samp_gives_schema_error(self):
        report = self._run("ks_2samp", "age_str", self._df())
        self._assert_schema_error(report, "ks_2samp", "categorical")

    def test_categorical_column_to_ttest_gives_schema_error(self):
        report = self._run("ttest", "age_str", self._df())
        self._assert_schema_error(report, "ttest", "categorical")

    def test_categorical_column_to_mannwhitney_gives_schema_error(self):
        report = self._run("mannwhitney", "age_str", self._df())
        self._assert_schema_error(report, "mannwhitney", "categorical")

    def test_categorical_column_to_levene_gives_schema_error(self):
        report = self._run("levene", "age_str", self._df())
        self._assert_schema_error(report, "levene", "categorical")

    def test_categorical_column_to_cramervonmises_gives_schema_error(self):
        report = self._run("cramervonmises", "age_str", self._df())
        self._assert_schema_error(report, "cramervonmises", "categorical")

    def test_categorical_column_to_wasserstein_gives_schema_error(self):
        report = self._run("wasserstein", "age_str", self._df())
        self._assert_schema_error(report, "wasserstein", "categorical")

    def test_categorical_column_to_mmd_gives_schema_error(self):
        report = self._run("mmd", "age_str", self._df())
        self._assert_schema_error(report, "mmd", "categorical")

    def test_numeric_column_to_chisquare_gives_schema_error(self):
        report = self._run("chisquare", "age", self._df())
        self._assert_schema_error(report, "chisquare", "numeric")

    def test_binary_column_accepted_by_ks_2samp(self):
        report = self._run("ks_2samp", "flag", self._df())
        assert len(report.errors) == 0
        assert len(report.results) == 1

    def test_binary_column_accepted_by_chisquare(self):
        report = self._run("chisquare", "flag", self._df())
        assert len(report.errors) == 0
        assert len(report.results) == 1

    def test_no_feature_name_skips_routing(self):
        from ayn_ml.core.schema import TabularSchema
        from ayn_ml.core.spec import MonitoringPlan
        from ayn_ml.runner import Runner

        plan = MonitoringPlan(
            name="t",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(),
            metrics=[MetricSpec(name="ks_2samp")],
        )
        report = Runner(strict=False).run(plan, self._df(), reference=self._df())
        # No routing check — error comes from compute() (missing feature_name)
        assert len(report.errors) == 1
        assert report.errors[0].metric_name == "ks_2samp"

    def test_psi_accepts_both_numeric_and_categorical(self):
        df = self._df()
        report_num = self._run("psi", "age", df)
        report_cat = self._run("psi", "age_str", df)
        # PSI has no restriction — neither run should produce a column-type routing error
        assert not any(e.error_type == "SchemaError" and "column type" in e.message for e in report_num.errors)
        assert not any(e.error_type == "SchemaError" and "column type" in e.message for e in report_cat.errors)

    def test_feature_types_declaration_overrides_dtype(self):
        from ayn_ml.core.schema import TabularSchema

        df = self._df()
        # 'age' is float (numeric by dtype), but declared categorical → chisquare routing passes
        schema = TabularSchema(feature_types={"age": "categorical"})
        report = self._run("chisquare", "age", df, schema=schema)
        # Routing passes; compute() may or may not fail (float→categorical conversion)
        # — the point is routing does NOT block it with a column-type mismatch error
        assert not any(e.error_type == "SchemaError" and "column type" in e.message for e in report.errors)

    def test_binary_nullable_integer_column_classified_as_binary(self):
        import pandas as pd

        from ayn_ml.core.schema import TabularSchema
        from ayn_ml.metrics.tabular._helpers import classify_columns

        # pandas nullable Int64 with NaN surfaces as float64 in numpy — must still classify as binary
        df = pd.DataFrame({"flag_nullable": pd.array([0, 1, None, 0, 1], dtype="Int64")})
        kinds = classify_columns(df, TabularSchema())
        from ayn_ml.core.schema import ColumnType

        assert kinds["flag_nullable"] == ColumnType.binary

    def test_equalized_odds_both_target_cols_incompatible_reports_all_errors(self):
        from ayn_ml.core.schema import TabularSchema

        # Both prediction_col and label_col are float (numeric) → both errors should appear
        schema = TabularSchema(prediction_col="y_pred_proba", label_col="y_pred_proba", protected_cols=["age_str"])
        report = self._run_fairness("equalized_odds", "age_str", schema)
        assert len(report.errors) == 1
        err = report.errors[0]
        assert err.error_type == "SchemaError"
        assert "schema.prediction_col" in err.message
        assert "schema.label_col" in err.message

    # --- accepted_target_types tests (schema-column routing) ---

    def _run_fairness(self, metric_name: str, feature_name: str, schema):
        from ayn_ml.core.spec import MonitoringPlan
        from ayn_ml.runner import Runner

        plan = MonitoringPlan(
            name="t",
            model_id="m",
            model_version="1",
            data_schema=schema,
            metrics=[MetricSpec(name=metric_name, feature_name=feature_name)],
        )
        return Runner(strict=False).run(plan, self._df())

    def test_demographic_parity_numeric_prediction_col_gives_schema_error(self):
        from ayn_ml.core.schema import TabularSchema

        schema = TabularSchema(prediction_col="y_pred_proba", protected_cols=["age_str"])
        report = self._run_fairness("demographic_parity", "age_str", schema)
        self._assert_schema_error(report, "demographic_parity", "schema.prediction_col")

    def test_disparate_impact_numeric_prediction_col_gives_schema_error(self):
        from ayn_ml.core.schema import TabularSchema

        schema = TabularSchema(prediction_col="y_pred_proba", protected_cols=["age_str"])
        report = self._run_fairness("disparate_impact", "age_str", schema)
        self._assert_schema_error(report, "disparate_impact", "schema.prediction_col")

    def test_equalized_odds_numeric_prediction_col_gives_schema_error(self):
        from ayn_ml.core.schema import TabularSchema

        schema = TabularSchema(prediction_col="y_pred_proba", label_col="y_true", protected_cols=["age_str"])
        report = self._run_fairness("equalized_odds", "age_str", schema)
        self._assert_schema_error(report, "equalized_odds", "schema.prediction_col")

    def test_equalized_odds_numeric_label_col_gives_schema_error(self):
        from ayn_ml.core.schema import TabularSchema

        schema = TabularSchema(prediction_col="y_pred", label_col="y_pred_proba", protected_cols=["age_str"])
        report = self._run_fairness("equalized_odds", "age_str", schema)
        self._assert_schema_error(report, "equalized_odds", "schema.label_col")

    def test_demographic_parity_binary_prediction_col_passes_routing(self):
        from ayn_ml.core.schema import TabularSchema

        schema = TabularSchema(prediction_col="y_pred", protected_cols=["age_str"])
        report = self._run_fairness("demographic_parity", "age_str", schema)
        assert not any("schema.prediction_col" in e.message for e in report.errors)

    def test_equalized_odds_binary_schema_cols_passes_routing(self):
        from ayn_ml.core.schema import TabularSchema

        schema = TabularSchema(prediction_col="y_pred", label_col="y_true", protected_cols=["age_str"])
        report = self._run_fairness("equalized_odds", "age_str", schema)
        assert not any("schema." in e.message for e in report.errors)


# ── Phase 2d: new distance drift metrics ──────────────────────────────────────


class TestDistanceMetrics:
    """Hellinger, Jensen-Shannon, TVD (histogram-based), and energy distance."""

    BOUNDED = ["hellinger", "jensenshannon", "tvd"]

    def test_numeric_value_in_unit_interval(self, df_current, df_reference, tabular_schema):
        for name in self.BOUNDED:
            r = get_metric(name).compute(df_current, df_reference, tabular_schema, drift_spec(name))
            assert 0.0 <= r.value <= 1.0, f"{name}: {r.value} not in [0, 1]"

    def test_categorical_value_in_unit_interval(self, df_current, df_reference, tabular_schema):
        for name in self.BOUNDED:
            r = get_metric(name).compute(
                df_current, df_reference, tabular_schema, drift_spec(name, feature="category")
            )
            assert 0.0 <= r.value <= 1.0, f"{name} on categorical: {r.value} not in [0, 1]"

    def test_identical_distributions_give_zero(self, df_current, tabular_schema):
        for name in self.BOUNDED:
            r = get_metric(name).compute(df_current, df_current, tabular_schema, drift_spec(name))
            assert r.value == pytest.approx(0.0, abs=1e-5), f"{name}: expected 0 on identical distributions"

    def test_requires_reference(self, df_current, tabular_schema):
        for name in self.BOUNDED:
            with pytest.raises(SchemaError):
                get_metric(name).compute(df_current, None, tabular_schema, drift_spec(name))

    def test_drift_detected_on_shifted_age(self, df_current, df_reference, tabular_schema):
        for name in self.BOUNDED:
            r = get_metric(name).compute(df_current, df_reference, tabular_schema, drift_spec(name))
            assert r.value > 0.0, f"{name}: expected non-zero distance on shifted distribution"

    def test_bins_param_accepted(self, df_current, df_reference, tabular_schema):
        for name in self.BOUNDED:
            r = get_metric(name).compute(
                df_current, df_reference, tabular_schema, drift_spec(name, params={"bins": 20})
            )
            assert 0.0 <= r.value <= 1.0

    def test_energy_distance_nonnegative(self, df_current, df_reference, tabular_schema):
        r = get_metric("energy_distance").compute(
            df_current, df_reference, tabular_schema, drift_spec("energy_distance")
        )
        assert r.value >= 0.0

    def test_energy_distance_zero_on_identical(self, df_current, tabular_schema):
        r = get_metric("energy_distance").compute(
            df_current, df_current, tabular_schema, drift_spec("energy_distance")
        )
        assert r.value == pytest.approx(0.0, abs=1e-6)

    def test_energy_distance_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError):
            get_metric("energy_distance").compute(
                df_current, None, tabular_schema, drift_spec("energy_distance")
            )


# ── Phase 2d: new p-value statistical test metrics ────────────────────────────


class TestNewStatisticalTests:
    """Anderson-Darling, Epps-Singleton, Fisher exact, G-test, Z-test proportions."""

    def test_anderson_darling_pvalue_in_unit_interval(self, df_current, df_reference, tabular_schema):
        r = get_metric("anderson_darling").compute(
            df_current, df_reference, tabular_schema, drift_spec("anderson_darling")
        )
        assert 0.0 <= r.value <= 1.0

    def test_anderson_darling_has_ad_statistic(self, df_current, df_reference, tabular_schema):
        r = get_metric("anderson_darling").compute(
            df_current, df_reference, tabular_schema, drift_spec("anderson_darling")
        )
        assert r.effect_size is not None
        assert r.effect_size_label == "ad_statistic"

    def test_anderson_darling_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError):
            get_metric("anderson_darling").compute(
                df_current, None, tabular_schema, drift_spec("anderson_darling")
            )

    def test_epps_singleton_pvalue_in_unit_interval(self, df_current, df_reference, tabular_schema):
        r = get_metric("epps_singleton").compute(
            df_current, df_reference, tabular_schema, drift_spec("epps_singleton")
        )
        assert 0.0 <= r.value <= 1.0

    def test_epps_singleton_has_es_statistic(self, df_current, df_reference, tabular_schema):
        r = get_metric("epps_singleton").compute(
            df_current, df_reference, tabular_schema, drift_spec("epps_singleton")
        )
        assert r.effect_size is not None
        assert r.effect_size_label == "es_statistic"

    def test_epps_singleton_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError):
            get_metric("epps_singleton").compute(
                df_current, None, tabular_schema, drift_spec("epps_singleton")
            )

    def test_fisher_exact_pvalue_in_unit_interval(self, df_current, df_reference, tabular_schema):
        r = get_metric("fisher_exact").compute(
            df_current, df_reference, tabular_schema, drift_spec("fisher_exact", feature="y_pred")
        )
        assert 0.0 <= r.value <= 1.0

    def test_fisher_exact_has_odds_ratio(self, df_current, df_reference, tabular_schema):
        r = get_metric("fisher_exact").compute(
            df_current, df_reference, tabular_schema, drift_spec("fisher_exact", feature="y_pred")
        )
        assert r.effect_size is not None
        assert r.effect_size_label == "odds_ratio"

    def test_fisher_exact_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError):
            get_metric("fisher_exact").compute(
                df_current, None, tabular_schema, drift_spec("fisher_exact", feature="y_pred")
            )

    def test_gtest_pvalue_in_unit_interval(self, df_current, df_reference, tabular_schema):
        r = get_metric("gtest").compute(
            df_current, df_reference, tabular_schema, drift_spec("gtest", feature="category")
        )
        assert 0.0 <= r.value <= 1.0

    def test_gtest_has_cramer_v(self, df_current, df_reference, tabular_schema):
        r = get_metric("gtest").compute(
            df_current, df_reference, tabular_schema, drift_spec("gtest", feature="category")
        )
        assert r.effect_size is not None
        assert r.effect_size_label == "cramer_v"

    def test_gtest_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError):
            get_metric("gtest").compute(
                df_current, None, tabular_schema, drift_spec("gtest", feature="category")
            )

    def test_gtest_raises_on_float_column(self, df_current, df_reference, tabular_schema):
        with pytest.raises(SchemaError, match="dtype"):
            get_metric("gtest").compute(
                df_current, df_reference, tabular_schema, drift_spec("gtest", feature="age")
            )

    def test_ztest_proportions_pvalue_in_unit_interval(self, df_current, df_reference, tabular_schema):
        r = get_metric("ztest_proportions").compute(
            df_current, df_reference, tabular_schema, drift_spec("ztest_proportions", feature="y_pred")
        )
        assert 0.0 <= r.value <= 1.0

    def test_ztest_proportions_has_z_score(self, df_current, df_reference, tabular_schema):
        r = get_metric("ztest_proportions").compute(
            df_current, df_reference, tabular_schema, drift_spec("ztest_proportions", feature="y_pred")
        )
        assert r.effect_size is not None
        assert r.effect_size_label == "z_score"

    def test_ztest_proportions_requires_reference(self, df_current, tabular_schema):
        with pytest.raises(SchemaError):
            get_metric("ztest_proportions").compute(
                df_current, None, tabular_schema, drift_spec("ztest_proportions", feature="y_pred")
            )

    def test_p_value_tests_detect_drift_on_shifted_age(self, df_current, df_reference, tabular_schema):
        """AD and ES should detect the deliberate age mean shift (ref=40, cur=50)."""
        for name in ["anderson_darling", "epps_singleton"]:
            r = get_metric(name).compute(
                df_current, df_reference, tabular_schema, drift_spec(name)
            )
            assert r.value < 0.05, f"{name}: expected p < 0.05 on deliberately shifted distribution"


# ── Phase 2e: new column-level statistics ─────────────────────────────────────


class TestColumnLevelStatistics:
    """sum, unique_count, in_range_count, out_range_count, in_list_count."""

    def test_sum_matches_numpy(self, df_current, tabular_schema):
        r = get_metric("sum").compute(df_current, None, tabular_schema, stat_spec("sum"))
        expected = float(np.sum(df_current["age"].to_numpy().astype(float)))
        assert r.value == pytest.approx(expected)

    def test_unique_count_on_category(self, df_current, tabular_schema):
        r = get_metric("unique_count").compute(
            df_current, None, tabular_schema, stat_spec("unique_count", feature="category")
        )
        assert r.value == 3  # "A", "B", "C"

    def test_in_range_count_default_bounds_counts_all(self, df_current, tabular_schema):
        r = get_metric("in_range_count").compute(df_current, None, tabular_schema, stat_spec("in_range_count"))
        assert r.value == len(df_current)

    def test_in_range_count_narrow_bounds(self, df_current, tabular_schema):
        s = stat_spec("in_range_count", params={"low": 40.0, "high": 60.0})
        r = get_metric("in_range_count").compute(df_current, None, tabular_schema, s)
        arr = df_current["age"].to_numpy()
        assert r.value == int(np.sum((arr >= 40.0) & (arr <= 60.0)))

    def test_in_and_out_range_are_complementary(self, df_current, tabular_schema):
        params = {"low": 40.0, "high": 60.0}
        in_r = get_metric("in_range_count").compute(
            df_current, None, tabular_schema, stat_spec("in_range_count", params=params)
        )
        out_r = get_metric("out_range_count").compute(
            df_current, None, tabular_schema, stat_spec("out_range_count", params=params)
        )
        assert in_r.value + out_r.value == len(df_current)

    def test_in_list_count_matches_numpy(self, df_current, tabular_schema):
        s = stat_spec("in_list_count", feature="category", params={"values": ["A", "B"]})
        r = get_metric("in_list_count").compute(df_current, None, tabular_schema, s)
        expected = int(np.sum(np.isin(df_current["category"].to_numpy(), ["A", "B"])))
        assert r.value == expected

    def test_in_list_count_empty_list_returns_zero(self, df_current, tabular_schema):
        r = get_metric("in_list_count").compute(
            df_current, None, tabular_schema, stat_spec("in_list_count", feature="category")
        )
        assert r.value == 0


# ── Phase 2e: new dataset-level statistics ────────────────────────────────────


class TestDatasetLevelStatistics:
    """row_count, column_count, almost_constant_columns, duplicate_rows, empty_columns."""

    def _ds_spec(self, name: str, **params) -> MetricSpec:
        return MetricSpec(name=name, feature_name=None, params=params)

    def test_row_count(self, df_current, tabular_schema):
        r = get_metric("row_count").compute(df_current, None, tabular_schema, self._ds_spec("row_count"))
        assert r.value == len(df_current)

    def test_column_count(self, df_current, tabular_schema):
        r = get_metric("column_count").compute(df_current, None, tabular_schema, self._ds_spec("column_count"))
        assert r.value == len(df_current.columns)

    def test_almost_constant_columns_zero_on_normal_data(self, df_current, tabular_schema):
        r = get_metric("almost_constant_columns").compute(
            df_current, None, tabular_schema, self._ds_spec("almost_constant_columns")
        )
        assert r.value == 0

    def test_almost_constant_columns_detects_constant(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x": [1] * 20, "y": list(range(20))})
        r = get_metric("almost_constant_columns").compute(
            df, None, tabular_schema, self._ds_spec("almost_constant_columns")
        )
        assert r.value == 1  # "x" is constant

    def test_almost_constant_columns_n_unique_param(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x": [1, 2] * 10, "y": list(range(20))})
        r = get_metric("almost_constant_columns").compute(
            df, None, tabular_schema, self._ds_spec("almost_constant_columns", n_unique=2)
        )
        assert r.value == 1  # "x" has exactly 2 unique values

    def test_duplicate_rows_zero_on_unique_data(self, df_current, tabular_schema):
        r = get_metric("duplicate_rows").compute(
            df_current, None, tabular_schema, self._ds_spec("duplicate_rows")
        )
        assert r.value == 0

    def test_duplicate_rows_counts_correctly(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        row = {"x": 1, "y": 2}
        df = pd.DataFrame([row, row, {"x": 3, "y": 4}])
        r = get_metric("duplicate_rows").compute(df, None, tabular_schema, self._ds_spec("duplicate_rows"))
        assert r.value == 1  # 3 total − 2 unique = 1 duplicate

    def test_empty_columns_zero_on_full_data(self, df_current, tabular_schema):
        r = get_metric("empty_columns").compute(
            df_current, None, tabular_schema, self._ds_spec("empty_columns")
        )
        assert r.value == 0

    def test_empty_columns_detects_null_column(self, tabular_schema):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [None, None, None]})
        r = get_metric("empty_columns").compute(df, None, tabular_schema, self._ds_spec("empty_columns"))
        assert r.value == 1
