"""Tests for CsvSource."""

from __future__ import annotations

import pytest

from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType, MonitoringPlan
from ayn_ml.data.csv import CsvSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan(**kwargs) -> MonitoringPlan:
    defaults = dict(
        name="p",
        model_id="m",
        model_version="1",
        data_schema=TabularSchema(proba_col=None, model_id_col=None, model_version_col=None),
        metrics=[],
    )
    defaults.update(kwargs)
    return MonitoringPlan(**defaults)


def _write_csv(path, content: str) -> None:
    path.write_text(content)


# ---------------------------------------------------------------------------
# Basic loading (backend="auto")
# ---------------------------------------------------------------------------


class TestCsvSourceLoad:
    def test_load_returns_dataframe(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred,age\n0,0,30\n1,1,40\n")
        result = CsvSource(path=csv_file).load(_plan())
        assert "y_true" in result.columns
        assert "y_pred" in result.columns

    def test_load_accepts_string_path(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n")
        result = CsvSource(path=str(csv_file)).load(_plan())
        assert "y_true" in result.columns

    def test_missing_file_raises_file_not_found(self, tmp_path):
        src = CsvSource(path=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError, match="nonexistent.csv"):
            src.load(_plan())

    def test_correct_row_count(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n0,1\n")
        assert len(CsvSource(path=csv_file).load(_plan())) == 3

    def test_header_only_csv_returns_zero_rows(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n")
        result = CsvSource(path=csv_file).load(_plan())
        assert len(result) == 0
        assert "y_true" in result.columns


# ---------------------------------------------------------------------------
# Column projection
# ---------------------------------------------------------------------------


class TestCsvSourceProjection:
    def test_projects_to_required_columns_only(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred,age,income\n0,0,30,50000\n1,1,40,60000\n")
        result = CsvSource(path=csv_file).load(_plan())
        assert "y_true" in result.columns
        assert "y_pred" in result.columns
        assert "income" not in result.columns

    def test_includes_feature_name_column(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred,age\n0,0,30\n1,1,40\n")
        plan = _plan(metrics=[MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="age")])
        assert "age" in CsvSource(path=csv_file).load(plan).columns

    def test_silently_skips_missing_required_columns(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n")
        plan = _plan(metrics=[MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="nonexistent")])
        result = CsvSource(path=csv_file).load(plan)
        assert "nonexistent" not in result.columns
        assert "y_true" in result.columns


# ---------------------------------------------------------------------------
# separator field — first-class, works across all backends
# ---------------------------------------------------------------------------


class TestCsvSourceSeparator:
    def test_pipe_separator(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true|y_pred\n0|0\n1|1\n")
        result = CsvSource(path=csv_file, separator="|").load(_plan())
        assert "y_true" in result.columns
        assert "y_pred" in result.columns

    def test_tab_separator(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true\ty_pred\n0\t0\n1\t1\n")
        result = CsvSource(path=csv_file, separator="\t").load(_plan())
        assert "y_true" in result.columns

    def test_separator_with_explicit_polars_backend(self, tmp_path):
        pytest.importorskip("polars")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true|y_pred\n0|0\n1|1\n")
        result = CsvSource(path=csv_file, backend="polars", separator="|").load(_plan())
        assert "y_true" in result.columns

    def test_separator_with_explicit_pandas_backend(self, tmp_path):
        pytest.importorskip("pandas")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true|y_pred\n0|0\n1|1\n")
        result = CsvSource(path=csv_file, backend="pandas", separator="|").load(_plan())
        assert "y_true" in result.columns


# ---------------------------------------------------------------------------
# Explicit backend="polars"
# ---------------------------------------------------------------------------


class TestCsvSourcePolarsBackend:
    def test_returns_polars_frame(self, tmp_path):
        pl = pytest.importorskip("polars")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred,age\n0,0,30\n1,1,40\n")
        plan = _plan(metrics=[MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="age")])
        result = CsvSource(path=csv_file, backend="polars").load(plan)
        assert isinstance(result, pl.DataFrame)
        assert "age" in result.columns

    def test_polars_native_kwarg_n_rows(self, tmp_path):
        """n_rows is the Polars-native way to limit rows."""
        pytest.importorskip("polars")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n0,1\n1,0\n")
        result = CsvSource(path=csv_file, backend="polars", read_kwargs={"n_rows": 2}).load(_plan())
        assert len(result) == 2

    def test_invalid_polars_kwarg_raises(self, tmp_path):
        """Passing a pandas-only kwarg to polars raises naturally from the backend."""
        pytest.importorskip("polars")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n")
        src = CsvSource(path=csv_file, backend="polars", read_kwargs={"nrows": 1})
        with pytest.raises(TypeError):
            src.load(_plan())


# ---------------------------------------------------------------------------
# Explicit backend="pandas"
# ---------------------------------------------------------------------------


class TestCsvSourcePandasBackend:
    def test_returns_pandas_frame(self, tmp_path):
        pd = pytest.importorskip("pandas")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred,age\n0,0,30\n1,1,40\n")
        plan = _plan(metrics=[MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="age")])
        result = CsvSource(path=csv_file, backend="pandas").load(plan)
        assert isinstance(result, pd.DataFrame)
        assert "age" in result.columns

    def test_pandas_native_kwarg_nrows(self, tmp_path):
        """nrows is the pandas-native way to limit rows."""
        pytest.importorskip("pandas")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n0,1\n1,0\n")
        result = CsvSource(path=csv_file, backend="pandas", read_kwargs={"nrows": 2}).load(_plan())
        assert len(result) == 2

    def test_pandas_used_even_when_polars_installed(self, tmp_path):
        """backend='pandas' bypasses Polars even when it is installed."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("polars")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n")
        result = CsvSource(path=csv_file, backend="pandas").load(_plan())
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# backend="auto" behaviour
# ---------------------------------------------------------------------------


class TestCsvSourceAutoBackend:
    def test_auto_prefers_polars(self, tmp_path):
        pl = pytest.importorskip("polars")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n")
        result = CsvSource(path=csv_file, backend="auto").load(_plan())
        assert isinstance(result, pl.DataFrame)

    def test_auto_falls_back_to_pandas(self, tmp_path, monkeypatch):
        pd = pytest.importorskip("pandas")
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n")

        import ayn_ml.data.csv as csv_module
        # Simulate polars not being installed by removing it from the preference order.
        monkeypatch.setattr(csv_module, "_AUTO_PREFERENCE", ("pandas",))

        result = CsvSource(path=csv_file).load(_plan())
        assert isinstance(result, pd.DataFrame)

    def test_auto_with_read_kwargs_emits_warning(self, tmp_path, caplog):
        import logging
        pytest.importorskip("polars")  # infer_schema_length is a Polars-only kwarg
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n")
        with caplog.at_level(logging.WARNING, logger="ayn_ml.data.csv"):
            CsvSource(path=csv_file, read_kwargs={"infer_schema_length": 100}).load(_plan())
        assert any("auto" in r.message and "read_kwargs" in r.message for r in caplog.records)

    def test_auto_without_read_kwargs_no_warning(self, tmp_path, caplog):
        import logging
        csv_file = tmp_path / "data.csv"
        _write_csv(csv_file, "y_true,y_pred\n0,0\n1,1\n")
        with caplog.at_level(logging.WARNING, logger="ayn_ml.data.csv"):
            CsvSource(path=csv_file).load(_plan())
        assert not caplog.records
