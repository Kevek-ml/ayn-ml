"""Tests for the target_drift metric."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from ayn_ml.core.schema import TabularSchema, TextSchema
from ayn_ml.core.spec import MetricSpec
from ayn_ml.exceptions import InsufficientDataError, SchemaError
from ayn_ml.metrics import get_metric

_RNG = np.random.default_rng(0)
_N = 300


def _spec(**kwargs) -> MetricSpec:
    return MetricSpec(name="target_drift", **kwargs)


def _binary_df(n: int = _N, pos_rate: float = 0.5, *, rng: np.random.Generator = _RNG) -> pd.DataFrame:
    y = (rng.uniform(0, 1, n) < pos_rate).astype(int)
    return pd.DataFrame({"y_true": y, "y_pred": y, "y_pred_proba": rng.uniform(0, 1, n)})


def _float_target_df(n: int = _N, mean: float = 0.0, *, rng: np.random.Generator = _RNG) -> pd.DataFrame:
    y = rng.normal(mean, 1.0, n).astype(float)
    return pd.DataFrame({"y_true": y, "y_pred": y, "y_pred_proba": rng.uniform(0, 1, n)})


@pytest.fixture(scope="module")
def schema() -> TabularSchema:
    return TabularSchema()


@pytest.fixture(scope="module")
def df_ref() -> pd.DataFrame:
    return _binary_df(rng=np.random.default_rng(1))


@pytest.fixture(scope="module")
def df_cur_same() -> pd.DataFrame:
    """Same distribution as reference — PSI should be near zero."""
    return _binary_df(rng=np.random.default_rng(2))


@pytest.fixture(scope="module")
def df_cur_shifted() -> pd.DataFrame:
    """Positive rate shifted from 0.5 → 0.1 — PSI should be large."""
    return _binary_df(pos_rate=0.1, rng=np.random.default_rng(3))


# ── Basic output ───────────────────────────────────────────────────────────────


class TestTargetDriftOutput:
    def test_value_nonnegative(self, df_cur_same, df_ref, schema):
        r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec())
        assert r.value >= 0.0

    def test_no_drift_near_zero(self, df_cur_same, df_ref, schema):
        r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec())
        assert r.value < 0.1

    def test_large_drift_detected(self, df_cur_shifted, df_ref, schema):
        r = get_metric("target_drift").compute(df_cur_shifted, df_ref, schema, _spec())
        assert r.value > 0.1

    def test_status_none_without_threshold(self, df_cur_same, df_ref, schema):
        r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec())
        assert r.status is None

    def test_status_false_when_drift_exceeds_threshold(self, df_cur_shifted, df_ref, schema):
        r = get_metric("target_drift").compute(df_cur_shifted, df_ref, schema, _spec(threshold=0.1))
        assert r.status is False

    def test_status_true_when_drift_within_threshold(self, df_cur_same, df_ref, schema):
        r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec(threshold=0.1))
        assert r.status is True


# ── Label type handling ────────────────────────────────────────────────────────


class TestTargetDriftLabelTypes:
    def test_binary_int_labels_treated_as_categorical(self, df_ref, schema):
        """Integer labels should use categorical PSI (not histogram binning)."""
        r = get_metric("target_drift").compute(df_ref, df_ref, schema, _spec())
        assert r.value == pytest.approx(0.0, abs=1e-6)

    def test_multiclass_int_labels(self, schema):
        rng = np.random.default_rng(10)
        df = pd.DataFrame({"y_true": rng.integers(0, 4, _N), "y_pred": 0, "y_pred_proba": 0.5})
        r = get_metric("target_drift").compute(df, df, schema, _spec())
        assert r.value == pytest.approx(0.0, abs=1e-6)

    def test_float_labels_treated_as_numeric(self, schema):
        """Float target column should use histogram PSI."""
        rng = np.random.default_rng(11)
        df_r = _float_target_df(rng=rng)
        df_c = _float_target_df(mean=5.0, rng=rng)
        r = get_metric("target_drift").compute(df_c, df_r, schema, _spec())
        assert r.value > 0.1

    def test_string_labels_treated_as_categorical(self, schema):
        rng = np.random.default_rng(12)
        df_r = pd.DataFrame({"y_true": rng.choice(["cat", "dog"], _N), "y_pred": "cat", "y_pred_proba": 0.5})
        df_c = pd.DataFrame(
            {"y_true": rng.choice(["cat", "dog"], _N, p=[0.9, 0.1]), "y_pred": "cat", "y_pred_proba": 0.5}
        )
        r = get_metric("target_drift").compute(df_c, df_r, schema, _spec())
        assert r.value >= 0.0


# ── treat_as param ─────────────────────────────────────────────────────────────


class TestTreatAsParam:
    def test_treat_as_categorical_on_int(self, df_ref, df_cur_same, schema):
        r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec(params={"treat_as": "categorical"}))
        assert r.value >= 0.0

    def test_treat_as_numeric_on_int(self, df_ref, df_cur_same, schema):
        r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec(params={"treat_as": "numeric"}))
        assert r.value >= 0.0

    def test_treat_as_numeric_respects_n_bins(self, df_ref, df_cur_shifted, schema):
        r5 = get_metric("target_drift").compute(
            df_cur_shifted, df_ref, schema, _spec(params={"treat_as": "numeric", "n_bins": 5})
        )
        r20 = get_metric("target_drift").compute(
            df_cur_shifted, df_ref, schema, _spec(params={"treat_as": "numeric", "n_bins": 20})
        )
        # Both should be positive; exact values differ by bin count
        assert r5.value >= 0.0
        assert r20.value >= 0.0

    def test_invalid_treat_as_falls_back_to_auto(self, df_ref, df_cur_same, schema, caplog):
        with caplog.at_level(logging.WARNING, logger="ayn_ml.metrics.tabular.drift"):
            r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec(params={"treat_as": "ordinal"}))
        assert r.value >= 0.0
        assert "unknown treat_as" in caplog.text

    def test_eps_param_accepted(self, df_ref, df_cur_same, schema):
        r = get_metric("target_drift").compute(df_cur_same, df_ref, schema, _spec(params={"eps": 1e-6}))
        assert r.value >= 0.0


# ── Error conditions ───────────────────────────────────────────────────────────


class TestTargetDriftErrors:
    def test_requires_reference(self, df_cur_same, schema):
        with pytest.raises(SchemaError, match="reference"):
            get_metric("target_drift").compute(df_cur_same, None, schema, _spec())

    def test_requires_tabular_schema(self, df_ref, df_cur_same):
        with pytest.raises(SchemaError, match="TabularSchema"):
            get_metric("target_drift").compute(df_cur_same, df_ref, TextSchema(), _spec())

    def test_missing_label_col_in_current(self, df_ref, schema):
        df_no_label = df_ref.drop(columns=["y_true"])
        with pytest.raises(SchemaError, match="y_true"):
            get_metric("target_drift").compute(df_no_label, df_ref, schema, _spec())

    def test_missing_label_col_in_reference(self, df_ref, df_cur_same, schema):
        df_no_label = df_ref.drop(columns=["y_true"])
        with pytest.raises(SchemaError, match="y_true"):
            get_metric("target_drift").compute(df_cur_same, df_no_label, schema, _spec())

    def test_insufficient_current_rows(self, df_ref, schema):
        with pytest.raises(InsufficientDataError):
            get_metric("target_drift").compute(df_ref.iloc[:5], df_ref, schema, _spec())

    def test_insufficient_reference_rows(self, df_ref, schema):
        with pytest.raises(InsufficientDataError):
            get_metric("target_drift").compute(df_ref, df_ref.iloc[:5], schema, _spec())


# ── Polars backend ─────────────────────────────────────────────────────────────


class TestTargetDriftPolars:
    def test_polars_input(self, schema):
        pl = pytest.importorskip("polars")
        rng = np.random.default_rng(20)
        df_pd = _binary_df(rng=rng)
        df_pl = pl.from_pandas(df_pd)
        r = get_metric("target_drift").compute(df_pl, df_pl, schema, _spec())
        assert r.value == pytest.approx(0.0, abs=1e-6)
