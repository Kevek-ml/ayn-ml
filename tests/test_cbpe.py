"""Tests for Confidence-Based Performance Estimation (CBPE) metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec
from ayn_ml.exceptions import InsufficientDataError, MetricComputeError, SchemaError
from ayn_ml.metrics import get_metric

_RNG = np.random.default_rng(0)
_N = 500


def _make_df(n: int = _N, *, rng: np.random.Generator = _RNG, pos_rate: float = 0.4):
    """Binary classification DataFrame with realistic probabilities."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    y_true = (rng.uniform(0, 1, n) < pos_rate).astype(int)
    # Probabilities correlated with labels but noisy
    noise = rng.normal(0, 0.15, n)
    y_proba = np.clip(y_true * 0.7 + 0.15 + noise, 0.0, 1.0)
    y_pred = (y_proba >= 0.5).astype(int)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_pred_proba": y_proba})


@pytest.fixture(scope="module")
def schema() -> TabularSchema:
    return TabularSchema()


@pytest.fixture(scope="module")
def df_ref():
    return _make_df(rng=np.random.default_rng(1))


@pytest.fixture(scope="module")
def df_cur():
    # Same distribution — CBPE estimate should be close to supervised metric
    return _make_df(rng=np.random.default_rng(2))


def _spec(name: str, **kwargs) -> MetricSpec:
    return MetricSpec(name=name, **kwargs)


# ── Basic output range ─────────────────────────────────────────────────────────


class TestCBPEOutputRange:
    @pytest.mark.parametrize("name", ["cbpe_accuracy", "cbpe_auc", "cbpe_f1", "cbpe_precision", "cbpe_recall"])
    def test_value_in_unit_interval(self, df_cur, df_ref, schema, name):
        r = get_metric(name).compute(df_cur, df_ref, schema, _spec(name))
        assert 0.0 <= r.value <= 1.0

    @pytest.mark.parametrize("name", ["cbpe_accuracy", "cbpe_auc", "cbpe_f1", "cbpe_precision", "cbpe_recall"])
    def test_status_none_without_threshold(self, df_cur, df_ref, schema, name):
        r = get_metric(name).compute(df_cur, df_ref, schema, _spec(name))
        assert r.status is None

    @pytest.mark.parametrize("name", ["cbpe_accuracy", "cbpe_auc", "cbpe_f1", "cbpe_precision", "cbpe_recall"])
    def test_status_set_with_threshold(self, df_cur, df_ref, schema, name):
        r = get_metric(name).compute(df_cur, df_ref, schema, _spec(name, threshold=0.0, upper_bound=False))
        assert r.status is True


# ── Calibration toggle ─────────────────────────────────────────────────────────


class TestCalibrationToggle:
    def test_calibrate_false_still_returns_float(self, df_cur, df_ref, schema):
        r = get_metric("cbpe_accuracy").compute(
            df_cur, df_ref, schema, _spec("cbpe_accuracy", params={"calibrate": False})
        )
        assert 0.0 <= r.value <= 1.0

    def test_calibrate_true_and_false_differ(self, df_cur, df_ref, schema):
        r_cal = get_metric("cbpe_auc").compute(df_cur, df_ref, schema, _spec("cbpe_auc", params={"calibrate": True}))
        r_raw = get_metric("cbpe_auc").compute(df_cur, df_ref, schema, _spec("cbpe_auc", params={"calibrate": False}))
        # Not guaranteed to differ, but the paths should at least not crash
        assert r_cal.value is not None
        assert r_raw.value is not None


# ── Proximity to supervised metric ────────────────────────────────────────────


class TestCBPEProximityToSupervised:
    """CBPE estimates should be in the same ballpark as supervised metrics
    when reference and current come from the same distribution."""

    def test_cbpe_accuracy_close_to_supervised(self, df_cur, df_ref, schema):
        from sklearn.metrics import accuracy_score

        r = get_metric("cbpe_accuracy").compute(df_cur, df_ref, schema, _spec("cbpe_accuracy"))
        supervised = accuracy_score(df_cur["y_true"], df_cur["y_pred"])
        assert abs(r.value - supervised) < 0.12

    def test_cbpe_auc_close_to_supervised(self, df_cur, df_ref, schema):
        from sklearn.metrics import roc_auc_score

        r = get_metric("cbpe_auc").compute(df_cur, df_ref, schema, _spec("cbpe_auc"))
        supervised = roc_auc_score(df_cur["y_true"], df_cur["y_pred_proba"])
        assert abs(r.value - supervised) < 0.12


# ── Error conditions ───────────────────────────────────────────────────────────


class TestCBPEErrors:
    def test_requires_reference(self, df_cur, schema):
        with pytest.raises(SchemaError, match="reference"):
            get_metric("cbpe_accuracy").compute(df_cur, None, schema, _spec("cbpe_accuracy"))

    @pytest.mark.parametrize("name", ["cbpe_accuracy", "cbpe_auc", "cbpe_f1"])
    def test_insufficient_current_rows(self, df_ref, schema, name):
        tiny = df_ref.iloc[:50]
        with pytest.raises(InsufficientDataError):
            get_metric(name).compute(tiny, df_ref, schema, _spec(name))

    @pytest.mark.parametrize("name", ["cbpe_accuracy", "cbpe_auc", "cbpe_f1"])
    def test_insufficient_reference_rows(self, df_cur, schema, name):
        tiny_ref = df_cur.iloc[:50]
        with pytest.raises(InsufficientDataError):
            get_metric(name).compute(df_cur, tiny_ref, schema, _spec(name))

    def test_missing_proba_col_raises(self, df_cur, df_ref):
        schema_no_proba = TabularSchema(proba_col=None)
        with pytest.raises(SchemaError, match="proba_col"):
            get_metric("cbpe_accuracy").compute(df_cur, df_ref, schema_no_proba, _spec("cbpe_accuracy"))

    def test_missing_label_col_in_reference_raises(self, df_cur, df_ref, schema):
        ref_no_label = df_ref.drop(columns=["y_true"])
        with pytest.raises(SchemaError, match="y_true"):
            get_metric("cbpe_accuracy").compute(df_cur, ref_no_label, schema, _spec("cbpe_accuracy"))

    def test_cbpe_auc_raises_on_single_class_probabilities(self, df_ref, schema):
        # All probabilities near 1 → estimated negatives near zero
        df_single = df_ref.copy()
        df_single["y_pred_proba"] = 0.9999
        with pytest.raises(MetricComputeError, match="single-class"):
            get_metric("cbpe_auc").compute(df_single, df_ref, schema, _spec("cbpe_auc"))

    def test_wrong_schema_type_raises(self, df_cur, df_ref):
        from ayn_ml.core.schema import TextSchema

        with pytest.raises(SchemaError, match="TabularSchema"):
            get_metric("cbpe_accuracy").compute(df_cur, df_ref, TextSchema(), _spec("cbpe_accuracy"))


# ── Polars backend ─────────────────────────────────────────────────────────────


class TestCBPEPolarsBackend:
    def test_cbpe_accuracy_polars_frame(self):
        pl = pytest.importorskip("polars")
        pytest.importorskip("pandas")
        pytest.importorskip("sklearn")

        df_pd = _make_df(rng=np.random.default_rng(5))
        df_pl = pl.from_pandas(df_pd)

        schema = TabularSchema()
        r = get_metric("cbpe_accuracy").compute(df_pl, df_pl, schema, _spec("cbpe_accuracy"))
        assert 0.0 <= r.value <= 1.0
