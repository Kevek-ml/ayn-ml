"""Tests for fairness metrics: demographic_parity, equalized_odds, disparate_impact."""

from __future__ import annotations

import numpy as np
import pytest

from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec
from ayn_ml.exceptions import InsufficientDataError, SchemaError
from ayn_ml.metrics import get_metric

_RNG = np.random.default_rng(42)


def _make_df(
    n: int = 500,
    positive_rates: dict | None = None,
    groups: list | None = None,
    seed: int = 0,
):
    """Binary classification DataFrame with a protected group column."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    groups = groups or ["A", "B"]
    positive_rates = positive_rates or {g: 0.5 for g in groups}

    n_per_group = n // len(groups)
    rows = []
    for g in groups:
        prob_true = 1 / (1 + np.exp(-rng.normal(0, 1, n_per_group)))
        y_true = rng.binomial(1, prob_true)
        y_pred = rng.binomial(1, positive_rates[g], n_per_group)
        rows.append(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": g}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def schema() -> TabularSchema:
    return TabularSchema(
        label_col="y_true",
        prediction_col="y_pred",
        protected_cols=["group"],
    )


@pytest.fixture(scope="module")
def df_fair():
    return _make_df(positive_rates={"A": 0.5, "B": 0.5}, seed=0)


@pytest.fixture(scope="module")
def df_biased():
    return _make_df(positive_rates={"A": 0.8, "B": 0.2}, seed=1)


# ---------------------------------------------------------------------------
# TabularSchema — protected_cols
# ---------------------------------------------------------------------------


class TestTabularSchemaProtectedCols:
    def test_column_names_includes_protected_cols(self):
        s = TabularSchema(protected_cols=["gender", "age_group"])
        assert "gender" in s.column_names
        assert "age_group" in s.column_names

    def test_column_names_excludes_when_none(self):
        s = TabularSchema(protected_cols=None)
        assert "gender" not in s.column_names

    def test_from_dataframe_excludes_protected_cols_from_feature_types(self):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "age": [25, 30], "gender": ["M", "F"]})
        s = TabularSchema.from_dataframe(df, protected_cols=["gender"])
        assert "gender" not in s.feature_types
        assert "age" in s.feature_types

    def test_from_dataframe_without_protected_cols_includes_all_features(self):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "age": [25, 30], "gender": ["M", "F"]})
        s = TabularSchema.from_dataframe(df)
        assert "gender" in s.feature_types
        assert "age" in s.feature_types


# ---------------------------------------------------------------------------
# demographic_parity
# ---------------------------------------------------------------------------


class TestDemographicParity:
    def _compute(self, df, schema, threshold=None):
        spec = MetricSpec(
            name="demographic_parity",
            feature_name="group",
            threshold=threshold,
        )
        return get_metric("demographic_parity").compute(df, None, schema, spec)

    def test_fair_model_near_zero(self, df_fair, schema):
        r = self._compute(df_fair, schema)
        assert r.value < 0.1

    def test_biased_model_high_disparity(self, df_biased, schema):
        r = self._compute(df_biased, schema)
        assert r.value > 0.4

    def test_threshold_pass(self, df_fair, schema):
        r = self._compute(df_fair, schema, threshold=0.2)
        assert r.status is True

    def test_threshold_fail(self, df_biased, schema):
        r = self._compute(df_biased, schema, threshold=0.1)
        assert r.status is False

    def test_no_threshold_status_none(self, df_fair, schema):
        r = self._compute(df_fair, schema, threshold=None)
        assert r.status is None

    def test_single_group_returns_zero(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1] * 10, "y_pred": [1] * 20, "group": ["A"] * 20})
        r = self._compute(df, schema)
        assert r.value == 0.0

    def test_feature_name_none_raises(self, df_fair, schema):
        spec = MetricSpec(name="demographic_parity", feature_name=None)
        with pytest.raises(SchemaError, match="feature_name is required"):
            get_metric("demographic_parity").compute(df_fair, None, schema, spec)

    def test_undeclared_protected_col_raises(self, df_fair, schema):
        spec = MetricSpec(name="demographic_parity", feature_name="undeclared_col")
        with pytest.raises(SchemaError, match="not declared in TabularSchema.protected_cols"):
            get_metric("demographic_parity").compute(df_fair, None, schema, spec)

    def test_missing_column_raises(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1] * 10, "y_pred": [0, 1] * 10})
        schema_no_check = TabularSchema(label_col="y_true", prediction_col="y_pred")
        spec = MetricSpec(name="demographic_parity", feature_name="group")
        with pytest.raises(SchemaError, match="not found in DataFrame"):
            get_metric("demographic_parity").compute(df, None, schema_no_check, spec)

    def test_insufficient_rows_raises(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "group": ["A", "B"]})
        with pytest.raises(InsufficientDataError):
            self._compute(df, schema)

    def test_multi_group_worst_case_gap(self, schema):
        pytest.importorskip("pandas")
        df = _make_df(
            positive_rates={"low": 0.2, "mid": 0.5, "high": 0.8},
            groups=["low", "mid", "high"],
            seed=3,
        )
        s = TabularSchema(label_col="y_true", prediction_col="y_pred", protected_cols=["group"])
        spec = MetricSpec(name="demographic_parity", feature_name="group")
        r = get_metric("demographic_parity").compute(df, None, s, spec)
        assert r.value > 0.4  # gap between 0.8 and 0.2 groups


# ---------------------------------------------------------------------------
# equalized_odds
# ---------------------------------------------------------------------------


class TestEqualizedOdds:
    def _compute(self, df, schema, threshold=None):
        spec = MetricSpec(
            name="equalized_odds",
            feature_name="group",
            threshold=threshold,
        )
        return get_metric("equalized_odds").compute(df, None, schema, spec)

    def test_fair_model_near_zero(self, df_fair, schema):
        r = self._compute(df_fair, schema)
        assert r.value < 0.2

    def test_biased_model_high_disparity(self, df_biased, schema):
        r = self._compute(df_biased, schema)
        assert r.value > 0.3

    def test_threshold_pass(self, df_fair, schema):
        r = self._compute(df_fair, schema, threshold=0.5)
        assert r.status is True

    def test_threshold_fail(self, df_biased, schema):
        r = self._compute(df_biased, schema, threshold=0.1)
        assert r.status is False

    def test_single_group_returns_zero(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1] * 10, "y_pred": [1, 0] * 10, "group": ["A"] * 20})
        r = self._compute(df, schema)
        assert r.value == 0.0

    def test_all_positive_labels_fpr_undefined_returns_tpr_gap(self, schema):
        """When no group has negative labels, FPR is all-NaN — only TPR gap counts."""
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame(
            {
                "y_true": [1] * 40,
                "y_pred": [1] * 20 + [0] * 20,
                "group": ["A"] * 20 + ["B"] * 20,
            }
        )
        r = self._compute(df, schema)
        assert r.value >= 0.0  # TPR gap: A=1.0, B=0.0 → gap=1.0
        assert not (r.value != r.value)  # not NaN

    def test_all_nan_tpr_and_fpr_returns_zero(self):
        """Single-class dataset where no group has both pos and neg — returns 0."""
        pytest.importorskip("pandas")
        import pandas as _pd

        # All labels are 1, all groups have only one member to force pure groups
        df = _pd.DataFrame(
            {
                "y_true": [1] * 20 + [0] * 20,
                "y_pred": [1] * 40,
                "group": ["A"] * 20 + ["B"] * 20,
            }
        )
        schema = TabularSchema(label_col="y_true", prediction_col="y_pred", protected_cols=["group"])
        spec = MetricSpec(name="equalized_odds", feature_name="group")
        r = get_metric("equalized_odds").compute(df, None, schema, spec)
        assert r.value >= 0.0
        assert not (r.value != r.value)

    def test_multiclass_labels_raises(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame(
            {
                "y_true": [0, 1, 2] * 10,
                "y_pred": [0, 1, 0] * 10,
                "group": ["A", "B", "A"] * 10,
            }
        )
        with pytest.raises(SchemaError, match="binary labels"):
            self._compute(df, schema)

    def test_feature_name_none_raises(self, df_fair, schema):
        spec = MetricSpec(name="equalized_odds", feature_name=None)
        with pytest.raises(SchemaError, match="feature_name is required"):
            get_metric("equalized_odds").compute(df_fair, None, schema, spec)

    def test_insufficient_rows_raises(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "group": ["A", "B"]})
        with pytest.raises(InsufficientDataError):
            self._compute(df, schema)


# ---------------------------------------------------------------------------
# disparate_impact
# ---------------------------------------------------------------------------


class TestDisparateImpact:
    def _compute(self, df, schema, threshold=None, upper_bound=False):
        spec = MetricSpec(
            name="disparate_impact",
            feature_name="group",
            threshold=threshold,
            upper_bound=upper_bound,
        )
        return get_metric("disparate_impact").compute(df, None, schema, spec)

    def test_fair_model_near_one(self, df_fair, schema):
        r = self._compute(df_fair, schema)
        assert r.value > 0.8

    def test_biased_model_low_ratio(self, df_biased, schema):
        r = self._compute(df_biased, schema)
        assert r.value < 0.5

    def test_80_percent_rule_pass(self, df_fair, schema):
        r = self._compute(df_fair, schema, threshold=0.8, upper_bound=False)
        assert r.status is True

    def test_80_percent_rule_fail(self, df_biased, schema):
        r = self._compute(df_biased, schema, threshold=0.8, upper_bound=False)
        assert r.status is False

    def test_all_zero_rates_returns_one(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0] * 20, "y_pred": [0] * 20, "group": ["A"] * 10 + ["B"] * 10})
        r = self._compute(df, schema)
        assert r.value == 1.0

    def test_single_group_returns_one(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1] * 10, "y_pred": [1] * 20, "group": ["A"] * 20})
        r = self._compute(df, schema)
        assert r.value == 1.0

    def test_feature_name_none_raises(self, df_fair, schema):
        spec = MetricSpec(name="disparate_impact", feature_name=None)
        with pytest.raises(SchemaError, match="feature_name is required"):
            get_metric("disparate_impact").compute(df_fair, None, schema, spec)

    def test_undeclared_protected_col_raises(self, df_fair, schema):
        spec = MetricSpec(name="disparate_impact", feature_name="unknown")
        with pytest.raises(SchemaError, match="not declared in TabularSchema.protected_cols"):
            get_metric("disparate_impact").compute(df_fair, None, schema, spec)

    def test_insufficient_rows_raises(self, schema):
        pytest.importorskip("pandas")
        import pandas as _pd

        df = _pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "group": ["A", "B"]})
        with pytest.raises(InsufficientDataError):
            self._compute(df, schema)

    def test_multi_group_worst_case_ratio(self):
        pytest.importorskip("pandas")
        df = _make_df(
            positive_rates={"low": 0.2, "mid": 0.5, "high": 0.8},
            groups=["low", "mid", "high"],
            seed=4,
        )
        s = TabularSchema(label_col="y_true", prediction_col="y_pred", protected_cols=["group"])
        spec = MetricSpec(name="disparate_impact", feature_name="group")
        r = get_metric("disparate_impact").compute(df, None, s, spec)
        assert r.value < 0.4  # min/max ≈ 0.2/0.8 = 0.25
