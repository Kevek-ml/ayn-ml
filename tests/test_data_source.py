import pandas as pd
import pytest

from ayn_ml.core.schema import TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType, MonitoringPlan
from ayn_ml.data.source import DataFrameSource, required_columns


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


@pytest.fixture
def df():
    return pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "age": [30, 40], "income": [50_000, 60_000]})


class TestDataFrameSource:
    def test_load_projects_to_schema_cols(self, df):
        plan = _plan()
        result = DataFrameSource(df).load(plan)
        assert "y_true" in result.columns
        assert "y_pred" in result.columns
        assert "income" not in result.columns

    def test_load_includes_metric_feature_name(self, df):
        plan = _plan(metrics=[MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="age")])
        result = DataFrameSource(df).load(plan)
        assert "age" in result.columns

    def test_missing_columns_silently_skipped(self, df):
        plan = _plan(metrics=[MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="nonexistent")])
        result = DataFrameSource(df).load(plan)
        assert "nonexistent" not in result.columns


class TestRequiredColumns:
    def test_always_includes_schema_cols(self):
        plan = _plan()
        cols = required_columns(plan)
        assert "y_true" in cols
        assert "y_pred" in cols

    def test_includes_metric_feature_name(self):
        plan = _plan(metrics=[MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="age")])
        cols = required_columns(plan)
        assert "age" in cols

    def test_excludes_optional_schema_cols_when_none(self):
        plan = _plan()  # proba_col=None, model_id_col=None, model_version_col=None
        cols = required_columns(plan)
        assert "y_pred_proba" not in cols
        assert "model_id" not in cols
        assert "model_version" not in cols

    def test_no_duplicates_when_same_feature_in_multiple_metrics(self):
        plan = _plan(
            metrics=[
                MetricSpec(name="psi", metric_type=MetricType.drift, feature_name="age"),
                MetricSpec(name="wasserstein", metric_type=MetricType.drift, feature_name="age"),
            ]
        )
        cols = required_columns(plan)
        assert cols.count("age") == 1

    def test_no_none_values(self):
        plan = _plan(metrics=[MetricSpec(name="accuracy", metric_type=MetricType.performance)])
        cols = required_columns(plan)
        assert None not in cols

    def test_includes_item_features_params(self):
        plan = _plan(
            metrics=[
                MetricSpec(
                    name="diversity",
                    metric_type=MetricType.recsys,
                    params={"item_features": ["f1", "f2"]},
                )
            ]
        )
        cols = required_columns(plan)
        assert "f1" in cols
        assert "f2" in cols

    def test_item_features_no_duplicates(self):
        plan = _plan(
            metrics=[
                MetricSpec(name="diversity", metric_type=MetricType.recsys,
                           params={"item_features": ["f1", "f2"]}),
                MetricSpec(name="serendipity", metric_type=MetricType.recsys,
                           params={"item_features": ["f1", "f3"]}),
            ]
        )
        cols = required_columns(plan)
        assert cols.count("f1") == 1
        assert "f2" in cols
        assert "f3" in cols
