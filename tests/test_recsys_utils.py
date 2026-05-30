"""Tests for recsys_utils.interactions_from_matrix."""

import pytest

pd = pytest.importorskip("pandas")

from ayn_ml.exceptions import SchemaError
from ayn_ml.metrics.recsys_utils import interactions_from_matrix


def _truth() -> pd.DataFrame:
    """2×3 binary relevance matrix with string index."""
    return pd.DataFrame(
        {"item_a": [1, 0], "item_b": [0, 1], "item_c": [1, 1]},
        index=["user_1", "user_2"],
    )


def _pred() -> pd.DataFrame:
    """2×3 score matrix matching _truth()."""
    return pd.DataFrame(
        {"item_a": [0.9, 0.2], "item_b": [0.3, 0.8], "item_c": [0.7, 0.6]},
        index=["user_1", "user_2"],
    )


class TestInteractionsFromMatrixPandas:
    def test_truth_only_columns(self):
        result = interactions_from_matrix(_truth())
        assert set(result.columns) == {"user_id", "item_id", "relevance"}

    def test_truth_only_row_count(self):
        # 2 users × 3 items = 6 rows
        assert len(interactions_from_matrix(_truth())) == 6

    def test_truth_only_user_values(self):
        result = interactions_from_matrix(_truth())
        assert set(result["user_id"].unique()) == {"user_1", "user_2"}

    def test_truth_only_item_values(self):
        result = interactions_from_matrix(_truth())
        assert set(result["item_id"].unique()) == {"item_a", "item_b", "item_c"}

    def test_truth_only_relevance_values(self):
        result = interactions_from_matrix(_truth())
        # user_1: item_a=1, item_b=0, item_c=1 → sum=2
        u1 = result[result["user_id"] == "user_1"]["relevance"].sum()
        assert u1 == 2

    def test_with_pred_adds_score_column(self):
        result = interactions_from_matrix(_truth(), _pred())
        assert "score" in result.columns

    def test_with_pred_row_count(self):
        assert len(interactions_from_matrix(_truth(), _pred())) == 6

    def test_with_pred_score_values(self):
        result = interactions_from_matrix(_truth(), _pred())
        u1_a = result[(result["user_id"] == "user_1") & (result["item_id"] == "item_a")]["score"].iloc[0]
        assert abs(u1_a - 0.9) < 1e-9

    def test_custom_column_names(self):
        result = interactions_from_matrix(
            _truth(), user_col="uid", item_col="iid", relevance_col="rating"
        )
        assert set(result.columns) == {"uid", "iid", "rating"}

    def test_custom_score_col_name(self):
        result = interactions_from_matrix(
            _truth(), _pred(), score_col="predicted"
        )
        assert "predicted" in result.columns

    def test_shape_mismatch_raises(self):
        bad_pred = _pred().iloc[:, :2]  # 2×2 vs 2×3
        with pytest.raises(SchemaError, match="shape"):
            interactions_from_matrix(_truth(), bad_pred)

    def test_column_mismatch_raises(self):
        bad_pred = _pred().rename(columns={"item_a": "item_x"})
        with pytest.raises(SchemaError, match="column"):
            interactions_from_matrix(_truth(), bad_pred)

    def test_index_mismatch_raises(self):
        bad_pred = _pred().copy()
        bad_pred.index = ["user_3", "user_4"]
        with pytest.raises(SchemaError, match="indices"):
            interactions_from_matrix(_truth(), bad_pred)

    def test_named_index_handled_correctly(self):
        # Index has an explicit name — reset_index would produce that name, not "index"
        truth = _truth().copy()
        truth.index.name = "custom_user"
        result = interactions_from_matrix(truth, user_col="user_id")
        assert "user_id" in result.columns
        assert set(result["user_id"].unique()) == {"user_1", "user_2"}

    def test_named_index_with_pred(self):
        truth = _truth().copy()
        pred = _pred().copy()
        truth.index.name = "uid"
        pred.index.name = "uid"
        result = interactions_from_matrix(truth, pred, user_col="user_id")
        assert "user_id" in result.columns
        assert "score" in result.columns

    def test_output_usable_by_precision_at_k(self):
        """Round-trip: matrix → interactions table → PrecisionAtK."""
        from ayn_ml.core.schema import RecSysSchema
        from ayn_ml.core.spec import MetricSpec
        from ayn_ml.metrics.recsys import PrecisionAtKMetric

        interactions = interactions_from_matrix(_truth(), _pred())
        schema = RecSysSchema()
        spec = MetricSpec(name="precision_at_k", params={"k": 2})
        result = PrecisionAtKMetric().compute(interactions, None, schema, spec)
        assert 0.0 <= result.value <= 1.0


class TestInteractionsFromMatrixPolars:
    def test_truth_only_columns(self):
        pl = pytest.importorskip("polars")
        truth = pl.from_pandas(_truth().reset_index().rename(columns={"index": "user_id"}))
        result = interactions_from_matrix(truth, user_col="user_id")
        assert set(result.columns) == {"user_id", "item_id", "relevance"}

    def test_truth_only_row_count(self):
        pl = pytest.importorskip("polars")
        truth = pl.from_pandas(_truth().reset_index().rename(columns={"index": "user_id"}))
        assert len(interactions_from_matrix(truth, user_col="user_id")) == 6

    def test_with_pred_adds_score_column(self):
        pl = pytest.importorskip("polars")
        truth = pl.from_pandas(_truth().reset_index().rename(columns={"index": "user_id"}))
        pred = pl.from_pandas(_pred().reset_index().rename(columns={"index": "user_id"}))
        result = interactions_from_matrix(truth, pred, user_col="user_id")
        assert "score" in result.columns

    def test_column_mismatch_raises(self):
        pl = pytest.importorskip("polars")
        truth = pl.from_pandas(_truth().reset_index().rename(columns={"index": "user_id"}))
        pred = pl.from_pandas(
            _pred().rename(columns={"item_a": "item_x"}).reset_index().rename(columns={"index": "user_id"})
        )
        with pytest.raises(SchemaError, match="column"):
            interactions_from_matrix(truth, pred, user_col="user_id")
