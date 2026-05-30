"""Tests for ExcelSource."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

# openpyxl is needed for _write_xlsx (test setup) in almost every test.
openpyxl = pytest.importorskip("openpyxl")
pd = pytest.importorskip("pandas")

from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType, MonitoringPlan
from ayn_ml.data.excel import ExcelSource

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


def _write_xlsx(path: Path, df: pd.DataFrame, sheet_name: str = "Sheet1") -> None:
    df.to_excel(path, sheet_name=sheet_name, index=False)


def _feature(name: str) -> MetricSpec:
    return MetricSpec(name="psi", metric_type=MetricType.drift, feature_name=name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 0, 1, 1],
            "age": [25, 35, 45, 55],
            "income": [30_000, 50_000, 70_000, 90_000],
        }
    )


@pytest.fixture
def xlsx_file(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    path = tmp_path / "data.xlsx"
    _write_xlsx(path, sample_df)
    return path


# ---------------------------------------------------------------------------
# Basic loading (backend="auto")
# ---------------------------------------------------------------------------


class TestExcelSourceLoad:
    def test_load_returns_dataframe(self, xlsx_file: Path) -> None:
        result = ExcelSource(path=xlsx_file).load(_plan())
        assert hasattr(result, "columns")

    def test_load_projects_to_schema_cols(self, xlsx_file: Path) -> None:
        result = ExcelSource(path=xlsx_file).load(_plan())
        assert "income" not in result.columns

    def test_load_includes_metric_feature_name(self, xlsx_file: Path) -> None:
        result = ExcelSource(path=xlsx_file).load(_plan(metrics=[_feature("age")])).columns
        assert "age" in result

    def test_load_missing_col_silently_skipped(self, xlsx_file: Path) -> None:
        result = ExcelSource(path=xlsx_file).load(_plan(metrics=[_feature("nonexistent")]))
        assert "nonexistent" not in result.columns

    def test_load_row_count(self, xlsx_file: Path, sample_df: pd.DataFrame) -> None:
        result = ExcelSource(path=xlsx_file).load(_plan(metrics=[_feature("age")]))
        assert len(result) == len(sample_df)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        src = ExcelSource(path=tmp_path / "nonexistent.xlsx")
        with pytest.raises(FileNotFoundError, match="nonexistent.xlsx"):
            src.load(_plan())

    def test_accepts_string_path(self, xlsx_file: Path) -> None:
        result = ExcelSource(path=str(xlsx_file)).load(_plan(metrics=[_feature("age")]))
        assert "age" in result.columns


# ---------------------------------------------------------------------------
# sheet_name — first-class field, works across both backends
# ---------------------------------------------------------------------------


class TestExcelSourceSheetName:
    def test_sheet_by_name(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        path = tmp_path / "named.xlsx"
        _write_xlsx(path, sample_df, sheet_name="predictions")
        result = ExcelSource(path=path, sheet_name="predictions").load(
            _plan(metrics=[_feature("age")])
        )
        assert "age" in result.columns
        assert len(result) == len(sample_df)

    def test_sheet_by_index_zero(self, xlsx_file: Path) -> None:
        result = ExcelSource(path=xlsx_file, sheet_name=0).load(_plan(metrics=[_feature("age")]))
        assert "age" in result.columns

    def test_sheet_by_index_nonzero(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        path = tmp_path / "two.xlsx"
        df2 = sample_df.assign(bonus=[1, 2, 3, 4])
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            sample_df.to_excel(writer, sheet_name="first", index=False)
            df2.to_excel(writer, sheet_name="second", index=False)
        result = ExcelSource(path=path, sheet_name=1).load(_plan(metrics=[_feature("bonus")]))
        assert "bonus" in result.columns

    def test_wrong_sheet_name_raises(self, xlsx_file: Path) -> None:
        with pytest.raises(Exception):
            ExcelSource(path=xlsx_file, sheet_name="does_not_exist").load(_plan())


# ---------------------------------------------------------------------------
# Explicit backend="polars"
# ---------------------------------------------------------------------------


class TestExcelSourcePolarsBackend:
    def test_returns_polars_frame(self, xlsx_file: Path) -> None:
        pl = pytest.importorskip("polars")
        pytest.importorskip("fastexcel")
        result = ExcelSource(path=xlsx_file, backend="polars").load(
            _plan(metrics=[_feature("age")])
        )
        assert isinstance(result, pl.DataFrame)
        assert "age" in result.columns

    def test_sheet_by_name(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        pytest.importorskip("polars")
        pytest.importorskip("fastexcel")
        path = tmp_path / "named.xlsx"
        _write_xlsx(path, sample_df, sheet_name="predictions")
        result = ExcelSource(path=path, backend="polars", sheet_name="predictions").load(
            _plan(metrics=[_feature("age")])
        )
        assert "age" in result.columns

    def test_sheet_by_index_zero_selects_first_sheet(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """sheet_name=0 must select the first sheet, not the second (tests 0→1 translation)."""
        pytest.importorskip("polars")
        pytest.importorskip("fastexcel")
        path = tmp_path / "two.xlsx"
        df2 = sample_df.assign(only_on_second=[10, 20, 30, 40])
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            sample_df.to_excel(writer, sheet_name="first", index=False)
            df2.to_excel(writer, sheet_name="second", index=False)
        result = ExcelSource(path=path, backend="polars", sheet_name=0).load(
            _plan(metrics=[_feature("age"), _feature("only_on_second")])
        )
        assert "age" in result.columns
        assert "only_on_second" not in result.columns  # second sheet column absent

    def test_sheet_by_index_nonzero(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        pytest.importorskip("polars")
        pytest.importorskip("fastexcel")
        path = tmp_path / "two.xlsx"
        df2 = sample_df.assign(bonus=[1, 2, 3, 4])
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            sample_df.to_excel(writer, sheet_name="first", index=False)
            df2.to_excel(writer, sheet_name="second", index=False)
        result = ExcelSource(path=path, backend="polars", sheet_name=1).load(
            _plan(metrics=[_feature("bonus")])
        )
        assert "bonus" in result.columns

    def test_missing_fastexcel_raises_with_hint(self, xlsx_file: Path) -> None:
        pytest.importorskip("polars")
        plan = _plan(metrics=[_feature("age")])
        src = ExcelSource(path=xlsx_file, backend="polars")
        with mock.patch.dict(__import__("sys").modules, {"fastexcel": None}):
            with pytest.raises(ImportError, match="pip install ayn-ml\\[excel\\]"):
                src.load(plan)


# ---------------------------------------------------------------------------
# Explicit backend="pandas"
# ---------------------------------------------------------------------------


class TestExcelSourcePandasBackend:
    def test_returns_pandas_frame(self, xlsx_file: Path) -> None:
        result = ExcelSource(path=xlsx_file, backend="pandas").load(
            _plan(metrics=[_feature("age")])
        )
        assert isinstance(result, pd.DataFrame)
        assert "age" in result.columns

    def test_pandas_used_even_when_polars_installed(self, xlsx_file: Path) -> None:
        pytest.importorskip("polars")
        result = ExcelSource(path=xlsx_file, backend="pandas").load(
            _plan(metrics=[_feature("age")])
        )
        assert isinstance(result, pd.DataFrame)

    def test_pandas_native_kwarg_usecols(self, tmp_path: Path) -> None:
        """usecols is a pandas-native kwarg for column pre-filtering."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        path = tmp_path / "kw.xlsx"
        _write_xlsx(path, df)
        result = ExcelSource(
            path=path,
            backend="pandas",
            read_kwargs={"usecols": ["col_a"]},
        ).load(_plan(metrics=[_feature("col_a")]))
        assert "col_a" in result.columns
        assert "col_b" not in result.columns

    def test_missing_openpyxl_raises_with_hint(self, xlsx_file: Path) -> None:
        plan = _plan(metrics=[_feature("age")])
        src = ExcelSource(path=xlsx_file, backend="pandas")
        with mock.patch.dict(__import__("sys").modules, {"openpyxl": None}):
            with pytest.raises(ImportError, match="pip install ayn-ml\\[excel\\]"):
                src.load(plan)


# ---------------------------------------------------------------------------
# backend="auto" behaviour
# ---------------------------------------------------------------------------


class TestExcelSourceAutoBackend:
    def test_auto_prefers_polars(self, xlsx_file: Path) -> None:
        pl = pytest.importorskip("polars")
        pytest.importorskip("fastexcel")
        result = ExcelSource(path=xlsx_file).load(_plan(metrics=[_feature("age")]))
        assert isinstance(result, pl.DataFrame)

    def test_auto_falls_back_to_pandas(self, xlsx_file: Path, monkeypatch) -> None:
        import ayn_ml.data.excel as excel_module
        monkeypatch.setattr(excel_module, "_AUTO_PREFERENCE", ("pandas",))
        result = ExcelSource(path=xlsx_file).load(_plan(metrics=[_feature("age")]))
        assert isinstance(result, pd.DataFrame)

    def test_auto_with_read_kwargs_emits_warning(self, xlsx_file: Path, caplog) -> None:
        import logging
        pytest.importorskip("polars")
        pytest.importorskip("fastexcel")
        with caplog.at_level(logging.WARNING, logger="ayn_ml.data.excel"):
            ExcelSource(path=xlsx_file, read_kwargs={"infer_schema_length": 100}).load(_plan())
        assert any("auto" in r.message and "read_kwargs" in r.message for r in caplog.records)

    def test_auto_without_read_kwargs_no_warning(self, xlsx_file: Path, caplog) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="ayn_ml.data.excel"):
            ExcelSource(path=xlsx_file).load(_plan())
        assert not caplog.records


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestExcelSourceImmutability:
    def test_is_frozen(self, xlsx_file: Path) -> None:
        src = ExcelSource(path=xlsx_file)
        with pytest.raises((TypeError, AttributeError)):
            src.path = "other.xlsx"  # type: ignore[misc]

    def test_read_kwargs_default_empty_dict(self, xlsx_file: Path) -> None:
        assert ExcelSource(path=xlsx_file).read_kwargs == {}

    def test_read_kwargs_not_shared_between_instances(self) -> None:
        s1 = ExcelSource(path="a.xlsx")
        s2 = ExcelSource(path="b.xlsx")
        assert s1.read_kwargs is not s2.read_kwargs
