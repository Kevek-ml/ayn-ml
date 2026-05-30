"""Tests for ayn_ml.metrics.tabular.profiler."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ayn_ml.core.schema import TabularSchema
from ayn_ml.metrics.tabular.profiler import profile_columns


def _schema(**kwargs) -> TabularSchema:
    return TabularSchema(**kwargs)


@pytest.fixture
def df_mixed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, 50),
            "flag": rng.integers(0, 2, 50),
            "region": [f"r{i % 4}" for i in range(50)],
        }
    )


class TestProfileColumnsNumeric:
    def test_numeric_keys_present(self, df_mixed):
        profile = profile_columns(df_mixed, ["age"], _schema())
        assert set(profile["age"]) == {"min", "max", "mean", "std", "p25", "p50", "p75", "null_count", "null_pct"}

    def test_numeric_values_are_finite(self, df_mixed):
        p = profile_columns(df_mixed, ["age"], _schema())["age"]
        for key in ("min", "max", "mean", "std", "p25", "p50", "p75"):
            assert isinstance(p[key], float)
            assert np.isfinite(p[key])

    def test_null_count_zero_when_no_nulls(self, df_mixed):
        p = profile_columns(df_mixed, ["age"], _schema())["age"]
        assert p["null_count"] == 0
        assert p["null_pct"] == 0.0

    def test_null_count_correct(self):
        df = pd.DataFrame({"x": [1.0, 2.0, None, None, 5.0]})
        p = profile_columns(df, ["x"], _schema())["x"]
        assert p["null_count"] == 2
        assert abs(p["null_pct"] - 0.4) < 1e-6

    def test_binary_column_uses_numeric_profile(self, df_mixed):
        p = profile_columns(df_mixed, ["flag"], _schema())
        assert "mean" in p["flag"]

    def test_ordering_p25_le_p50_le_p75(self, df_mixed):
        p = profile_columns(df_mixed, ["age"], _schema())["age"]
        assert p["p25"] <= p["p50"] <= p["p75"]


class TestProfileColumnsCategorical:
    def test_categorical_keys_present(self, df_mixed):
        profile = profile_columns(df_mixed, ["region"], _schema())
        assert set(profile["region"]) == {"null_count", "null_pct", "n_unique", "top_category"}

    def test_n_unique_correct(self, df_mixed):
        assert profile_columns(df_mixed, ["region"], _schema())["region"]["n_unique"] == 4

    def test_top_category_is_string(self, df_mixed):
        top = profile_columns(df_mixed, ["region"], _schema())["region"]["top_category"]
        assert isinstance(top, str)

    def test_null_count_for_categorical(self):
        df = pd.DataFrame({"cat": ["a", None, "b", None, "a"]})
        p = profile_columns(df, ["cat"], _schema())["cat"]
        assert p["null_count"] == 2


class TestProfileColumnsEdgeCases:
    def test_absent_column_logged_and_skipped(self, df_mixed):
        profile = profile_columns(df_mixed, ["age", "nonexistent"], _schema())
        assert "age" in profile
        assert "nonexistent" not in profile

    def test_empty_frame_returns_none_stats(self):
        df = pd.DataFrame({"age": pd.Series([], dtype=float)})
        p = profile_columns(df, ["age"], _schema())["age"]
        assert p["null_count"] == 0
        for key in ("min", "max", "mean", "std", "p25", "p50", "p75"):
            assert p[key] is None

    def test_all_null_numeric_returns_none_stats(self):
        df = pd.DataFrame({"x": [float("nan"), float("nan"), float("nan")]})
        p = profile_columns(df, ["x"], _schema())["x"]
        assert p["null_count"] == 3
        for key in ("min", "max", "mean", "std"):
            assert p[key] is None

    def test_all_null_categorical_top_is_none(self):
        df = pd.DataFrame({"cat": [None, None]})
        p = profile_columns(df, ["cat"], _schema())["cat"]
        assert p["top_category"] is None
        assert p["n_unique"] == 0

    def test_feature_types_override_respected(self):
        # integer column declared categorical → categorical profile shape
        df = pd.DataFrame({"encoded": [0, 1, 0, 2, 1]})
        schema = _schema(feature_types={"encoded": "categorical"})
        p = profile_columns(df, ["encoded"], schema)["encoded"]
        assert "n_unique" in p
        assert "mean" not in p

    def test_no_double_conversion(self, df_mixed):
        # Pass a narwhals frame directly — classify_columns must accept it without error
        import narwhals as nw

        native = nw.from_native(df_mixed, eager_only=True)
        profile = profile_columns(native, ["age", "region"], _schema())
        assert "age" in profile
        assert "region" in profile

    def test_single_element_numeric_std_is_none(self):
        df = pd.DataFrame({"x": [42.0]})
        p = profile_columns(df, ["x"], _schema())["x"]
        assert p["std"] is None
        assert p["mean"] == 42.0
        assert p["min"] == 42.0


class TestRunnerProfiling:
    def _plan(self, **kwargs):
        import ayn_ml.metrics  # noqa: F401
        from ayn_ml.core.spec import MonitoringPlan

        return MonitoringPlan(
            name="t",
            model_id="m",
            model_version="1",
            data_schema=TabularSchema(label_col="y_true", prediction_col="y_pred"),
            **kwargs,
        )

    def _df(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        n = 50
        return pd.DataFrame(
            {
                "y_true": rng.integers(0, 2, n),
                "y_pred": rng.integers(0, 2, n),
                "age": rng.normal(40, 10, n),
                "region": [f"r{i % 3}" for i in range(n)],
            }
        )

    def test_profiling_disabled_by_default(self):
        from ayn_ml.core.spec import MetricSpec
        from ayn_ml.runner import Runner

        plan = self._plan(
            enable_profiling=False,
            metrics=[MetricSpec(name="mean", feature_name="age")],
        )
        report = Runner(strict=False).run(plan, self._df())
        assert report.profile is None

    def test_profiling_enabled_covers_feature_and_target_cols(self):
        from ayn_ml.core.spec import MetricSpec
        from ayn_ml.runner import Runner

        plan = self._plan(
            enable_profiling=True,
            metrics=[MetricSpec(name="mean", feature_name="age")],
        )
        report = Runner(strict=False).run(plan, self._df())
        assert report.profile is not None
        assert "age" in report.profile  # feature col
        assert "y_pred" in report.profile  # prediction_col
        assert "y_true" in report.profile  # label_col

    def test_profile_in_to_dict_when_enabled(self):
        from ayn_ml.core.spec import MetricSpec
        from ayn_ml.runner import Runner

        plan = self._plan(
            enable_profiling=True,
            metrics=[MetricSpec(name="mean", feature_name="age")],
        )
        report = Runner(strict=False).run(plan, self._df())
        d = report.to_dict()
        assert "profile" in d
        assert "age" in d["profile"]

    def test_profile_absent_from_to_dict_when_disabled(self):
        from ayn_ml.core.spec import MetricSpec
        from ayn_ml.runner import Runner

        plan = self._plan(
            enable_profiling=False,
            metrics=[MetricSpec(name="mean", feature_name="age")],
        )
        report = Runner(strict=False).run(plan, self._df())
        assert "profile" not in report.to_dict()


class TestProfileColumnsPolars:
    """Integration tests using a Polars DataFrame as input."""

    @pytest.fixture(autouse=True)
    def _require_polars(self):
        pytest.importorskip("polars")

    def _df(self):
        import polars as pl

        rng = np.random.default_rng(42)
        return pl.DataFrame(
            {
                "age": rng.normal(40, 10, 30).tolist(),
                "region": [f"r{i % 3}" for i in range(30)],
            }
        )

    def test_numeric_profile_from_polars_frame(self):
        p = profile_columns(self._df(), ["age"], _schema())["age"]
        assert set(p) == {"min", "max", "mean", "std", "p25", "p50", "p75", "null_count", "null_pct"}
        assert np.isfinite(p["mean"])

    def test_categorical_profile_from_polars_frame(self):
        p = profile_columns(self._df(), ["region"], _schema())["region"]
        assert p["n_unique"] == 3
        assert isinstance(p["top_category"], str)

    def test_both_columns_from_polars_frame(self):
        profile = profile_columns(self._df(), ["age", "region"], _schema())
        assert "age" in profile
        assert "region" in profile
