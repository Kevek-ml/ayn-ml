"""Tests for recsys metrics (all 14): precision, recall, fbeta, hit_rate, map, ndcg, mrr,
diversity, novelty, popularity_bias, personalization, item_bias, user_bias, serendipity."""

import pytest

pd = pytest.importorskip("pandas")

from ayn_ml.core.schema import RecSysSchema, TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import SchemaError
from ayn_ml.metrics.recsys import (
    DiversityMetric,
    FBetaAtKMetric,
    HitRateMetric,
    ItemBiasMetric,
    MAPAtKMetric,
    MRRAtKMetric,
    NDCGAtKMetric,
    NoveltyMetric,
    PersonalizationMetric,
    PopularityBiasMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
    SerendipityMetric,
    UserBiasMetric,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _schema(**kwargs) -> RecSysSchema:
    return RecSysSchema(**kwargs)


def _spec(name: str, **params) -> MetricSpec:
    return MetricSpec(name=name, params=params)


def _interactions_df() -> pd.DataFrame:
    """Two users, five items each with scores and binary relevance.

    user_1 relevant items: {i1, i2, i3}; scores: i1>i2>i3>i4>i5
    user_2 relevant items: {i4, i5};     scores: i5>i4>i3>i2>i1

    At k=3:
      user_1 top3: [i1, i2, i3]  → hits=3
      user_2 top3: [i5, i4, i3]  → hits=2
    """
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i1", "i2", "i3", "i4", "i5"],
            "relevance": [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            "score": [0.9, 0.8, 0.7, 0.4, 0.2, 0.1, 0.2, 0.3, 0.6, 0.8],
        }
    )


def _rank_df() -> pd.DataFrame:
    """Same topology as ``_interactions_df`` but with explicit rank positions."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i1", "i2", "i3", "i4", "i5"],
            "relevance": [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            "rank": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        }
    )


# ---------------------------------------------------------------------------
# PrecisionAtKMetric
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_metric_type(self):
        assert PrecisionAtKMetric.metric_type == MetricType.recsys

    def test_requires_no_reference(self):
        assert PrecisionAtKMetric.requires_reference is False

    def test_k3_expected_value(self):
        # user_1: 3/3=1.0  user_2: 2/3  mean=5/6
        result = PrecisionAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("precision_at_k", k=3)
        )
        assert abs(result.value - 5 / 6) < 1e-5

    def test_k1_perfect_precision(self):
        # top-1 for both users is a relevant item → precision=1.0
        result = PrecisionAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("precision_at_k", k=1)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_default_k_is_10(self):
        # 10 items per user, all items covered → full hit for user_1, partial for user_2
        result = PrecisionAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("precision_at_k")
        )
        assert 0.0 <= result.value <= 1.0

    def test_no_relevant_items_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.5],
            }
        )
        result = PrecisionAtKMetric().compute(df, None, _schema(), _spec("precision_at_k", k=1))
        assert result.value == 0.0

    def test_rank_mode(self):
        schema = RecSysSchema(score_col="rank", recommendations_type="rank")
        result = PrecisionAtKMetric().compute(
            _rank_df(), None, schema, _spec("precision_at_k", k=3)
        )
        assert abs(result.value - 5 / 6) < 1e-5

    def test_graded_relevance_with_threshold(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3"],
                "relevance": [5, 3, 1],
                "score": [0.9, 0.8, 0.3],
            }
        )
        # threshold=2 → relevant={i1,i2}; top2=[i1,i2] → precision=1.0
        result = PrecisionAtKMetric().compute(
            df, None, _schema(), _spec("precision_at_k", k=2, relevance_threshold=2)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_empty_dataframe_raises(self):
        from ayn_ml.exceptions import InsufficientDataError

        df = pd.DataFrame({"user_id": [], "item_id": [], "relevance": [], "score": []})
        with pytest.raises(InsufficientDataError):
            PrecisionAtKMetric().compute(df, None, _schema(), _spec("precision_at_k", k=3))

    def test_score_col_none_uses_dataframe_order(self):
        # Without a score column, items are taken in row order: [i1, i2] for user_1
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3"],
                "relevance": [1, 0, 1],
            }
        )
        schema = RecSysSchema(score_col=None)
        # k=1 → top1=[i1], relevant={i1,i3} → precision=1.0
        result = PrecisionAtKMetric().compute(df, None, schema, _spec("precision_at_k", k=1))
        assert abs(result.value - 1.0) < 1e-5

    def test_k_larger_than_catalogue(self):
        # user has 3 items but k=5 → top_k has 3 entries, denominator is k=5
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3"],
                "relevance": [1, 1, 0],
                "score": [0.9, 0.8, 0.3],
            }
        )
        result = PrecisionAtKMetric().compute(df, None, _schema(), _spec("precision_at_k", k=5))
        # hits=2, denominator=5 → precision=2/5=0.4
        assert abs(result.value - 2 / 5) < 1e-5

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError, match="RecSysSchema"):
            PrecisionAtKMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("precision_at_k", k=3)
            )

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k"):
            PrecisionAtKMetric().compute(
                _interactions_df(), None, _schema(), _spec("precision_at_k", k=0)
            )

    def test_k_bool_raises(self):
        with pytest.raises(ValueError, match="k"):
            PrecisionAtKMetric().compute(
                _interactions_df(), None, _schema(), _spec("precision_at_k", k=True)
            )

    def test_missing_column_raises(self):
        df = _interactions_df().drop(columns=["relevance"])
        with pytest.raises(SchemaError):
            PrecisionAtKMetric().compute(df, None, _schema(), _spec("precision_at_k", k=3))

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = PrecisionAtKMetric().compute(df, None, _schema(), _spec("precision_at_k", k=3))
        assert abs(result.value - 5 / 6) < 1e-5

    def test_result_value_rounded_to_6_decimals(self):
        result = PrecisionAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("precision_at_k", k=3)
        )
        assert result.value == round(result.value, 6)

    def test_threshold_applied(self):
        result = PrecisionAtKMetric().compute(
            _interactions_df(),
            None,
            _schema(),
            MetricSpec(name="precision_at_k", params={"k": 3}, threshold=0.7, upper_bound=False),
        )
        assert result.status is not None


# ---------------------------------------------------------------------------
# RecallAtKMetric
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_metric_type(self):
        assert RecallAtKMetric.metric_type == MetricType.recsys

    def test_k3_perfect_recall(self):
        # user_1: 3/3=1.0  user_2: 2/2=1.0  mean=1.0
        result = RecallAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("recall_at_k", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_k1_partial_recall(self):
        # user_1: 1/3  user_2: 1/2  mean=5/12
        result = RecallAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("recall_at_k", k=1)
        )
        assert abs(result.value - 5 / 12) < 1e-5

    def test_no_relevant_items_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.5],
            }
        )
        result = RecallAtKMetric().compute(df, None, _schema(), _spec("recall_at_k", k=1))
        assert result.value == 0.0

    def test_rank_mode(self):
        schema = RecSysSchema(score_col="rank", recommendations_type="rank")
        result = RecallAtKMetric().compute(
            _rank_df(), None, schema, _spec("recall_at_k", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError, match="RecSysSchema"):
            RecallAtKMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("recall_at_k", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = RecallAtKMetric().compute(df, None, _schema(), _spec("recall_at_k", k=3))
        assert abs(result.value - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# FBetaAtKMetric
# ---------------------------------------------------------------------------


class TestFBetaAtK:
    def test_metric_type(self):
        assert FBetaAtKMetric.metric_type == MetricType.recsys

    def test_f1_at_k3(self):
        # user_1: p=1.0, r=1.0 → f1=1.0
        # user_2: p=2/3, r=1.0 → f1=0.8
        # mean=0.9
        result = FBetaAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("fbeta_at_k", k=3, beta=1.0)
        )
        assert abs(result.value - 0.9) < 1e-5

    def test_default_beta_is_1(self):
        result_explicit = FBetaAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("fbeta_at_k", k=3, beta=1.0)
        )
        result_default = FBetaAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("fbeta_at_k", k=3)
        )
        assert abs(result_explicit.value - result_default.value) < 1e-9

    def test_beta2_weights_recall(self):
        # beta=2 weights recall more; recall is already perfect at k=3, so
        # mean should be higher than beta=0.5
        result_b2 = FBetaAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("fbeta_at_k", k=3, beta=2.0)
        )
        result_b05 = FBetaAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("fbeta_at_k", k=3, beta=0.5)
        )
        assert result_b2.value >= result_b05.value

    def test_no_relevant_items_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.5],
            }
        )
        result = FBetaAtKMetric().compute(df, None, _schema(), _spec("fbeta_at_k", k=1))
        assert result.value == 0.0

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            FBetaAtKMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("fbeta_at_k", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = FBetaAtKMetric().compute(df, None, _schema(), _spec("fbeta_at_k", k=3))
        assert abs(result.value - 0.9) < 1e-5


# ---------------------------------------------------------------------------
# HitRateMetric
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_metric_type(self):
        assert HitRateMetric.metric_type == MetricType.recsys

    def test_k3_all_users_hit(self):
        result = HitRateMetric().compute(
            _interactions_df(), None, _schema(), _spec("hit_rate", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_k1_all_users_hit(self):
        # top-1 for both users is relevant → hit_rate=1.0
        result = HitRateMetric().compute(
            _interactions_df(), None, _schema(), _spec("hit_rate", k=1)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_partial_hit_rate(self):
        # user_1 relevant={i1,i2,i3}; user_2 has no relevant items
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i3", "i4"],
                "relevance": [1, 1, 0, 0],
                "score": [0.9, 0.8, 0.7, 0.3],
            }
        )
        result = HitRateMetric().compute(df, None, _schema(), _spec("hit_rate", k=1))
        assert abs(result.value - 0.5) < 1e-5

    def test_no_hits_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.5],
            }
        )
        result = HitRateMetric().compute(df, None, _schema(), _spec("hit_rate", k=1))
        assert result.value == 0.0

    def test_rank_mode(self):
        schema = RecSysSchema(score_col="rank", recommendations_type="rank")
        result = HitRateMetric().compute(
            _rank_df(), None, schema, _spec("hit_rate", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            HitRateMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("hit_rate", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = HitRateMetric().compute(df, None, _schema(), _spec("hit_rate", k=3))
        assert abs(result.value - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# MAPAtKMetric
# ---------------------------------------------------------------------------


class TestMAPAtK:
    """MAP@K tests.

    Fixture (k=3):
      user_1: top3=[i1,i2,i3], relevant={i1,i2,i3}
        hits at positions 1,2,3 → P@1=1, P@2=1, P@3=1
        AP@3 = (1+1+1)/3 / 3 * 3 = 3/3 = 1.0
        (AP = precision_sum / |relevant| = 3/3 = 1.0)
      user_2: top3=[i5,i4,i3], relevant={i4,i5}
        hits at positions 1,2 → P@1=1, P@2=1
        AP@3 = (1+1)/2 = 1.0
      mean MAP@3 = 1.0
    """

    def test_metric_type(self):
        assert MAPAtKMetric.metric_type == MetricType.recsys

    def test_requires_no_reference(self):
        assert MAPAtKMetric.requires_reference is False

    def test_k3_expected_value(self):
        result = MAPAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("map_at_k", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_k1_expected_value(self):
        # user_1: top1=[i1]∈relevant → AP = (1/1)/3 = 1/3
        # user_2: top1=[i5]∈relevant → AP = (1/1)/2 = 1/2
        # mean = (1/3 + 1/2)/2 = 5/12
        result = MAPAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("map_at_k", k=1)
        )
        assert abs(result.value - 5 / 12) < 1e-5

    def test_partial_hit(self):
        # user_1: top3=[i1,i2,i3], relevant={i1,i3}
        # hits at positions 1,3 → precision_sum = 1/1 + 2/3 = 5/3
        # AP = (5/3) / 2 = 5/6
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3", "i4", "i5"],
                "relevance": [1, 0, 1, 0, 0],
                "score": [0.9, 0.8, 0.7, 0.4, 0.2],
            }
        )
        result = MAPAtKMetric().compute(df, None, _schema(), _spec("map_at_k", k=3))
        assert abs(result.value - 5 / 6) < 1e-5

    def test_more_relevant_than_k_normalizes_by_full_set(self):
        # 5 relevant items, k=2, top2 are both hits
        # AP = (1/1 + 2/2) / 5 = 2/5 = 0.4  (NOT 1.0 — normalized by |relevant|=5)
        df = pd.DataFrame(
            {
                "user_id": ["u1"] * 5,
                "item_id": ["i1", "i2", "i3", "i4", "i5"],
                "relevance": [1, 1, 1, 1, 1],
                "score": [0.9, 0.8, 0.3, 0.2, 0.1],
            }
        )
        result = MAPAtKMetric().compute(df, None, _schema(), _spec("map_at_k", k=2))
        assert abs(result.value - 2 / 5) < 1e-5

    def test_no_relevant_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.5],
            }
        )
        result = MAPAtKMetric().compute(df, None, _schema(), _spec("map_at_k", k=2))
        assert result.value == 0.0

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            MAPAtKMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("map_at_k", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = MAPAtKMetric().compute(df, None, _schema(), _spec("map_at_k", k=3))
        assert abs(result.value - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# NDCGAtKMetric
# ---------------------------------------------------------------------------


class TestNDCGAtK:
    """NDCG@K tests.

    Binary relevance fixture (k=3):
      user_1: top3=[i1,i2,i3], all relevant
        DCG  = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.631 + 0.5 = 2.131
        IDCG = same (ideal order) = 2.131  → NDCG = 1.0
      user_2: top3=[i5,i4,i3], relevant={i4,i5}
        DCG  = 1/log2(2) + 1/log2(3) + 0 = 1 + 0.631 = 1.631
        IDCG = 1/log2(2) + 1/log2(3) = 1.631  → NDCG = 1.0
      mean NDCG@3 = 1.0
    """

    def test_metric_type(self):
        assert NDCGAtKMetric.metric_type == MetricType.recsys

    def test_requires_no_reference(self):
        assert NDCGAtKMetric.requires_reference is False

    def test_k3_perfect_ndcg(self):
        result = NDCGAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("ndcg_at_k", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_suboptimal_ranking(self):
        # user_1: relevant=[i3,i2,i1] (worst first), scores reversed
        # top3=[i5,i4,i3], relevant={i1,i2,i3} → only i3 hit at rank 3
        # DCG = 0+0+1/log2(4) = 0.5
        # IDCG = 1/log2(2)+1/log2(3)+1/log2(4) = 1+0.631+0.5 = 2.131
        # NDCG = 0.5/2.131 ≈ 0.2347
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3", "i4", "i5"],
                "relevance": [1, 1, 1, 0, 0],
                "score": [0.2, 0.3, 0.7, 0.8, 0.9],  # non-relevant scored highest
            }
        )
        result = NDCGAtKMetric().compute(df, None, _schema(), _spec("ndcg_at_k", k=3))
        import math
        # ranks 1,2,3 (1-based): denominator = log2(rank+1)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, 4))
        dcg = 1.0 / math.log2(3 + 1)  # i3 hits at rank 3
        assert abs(result.value - dcg / idcg) < 1e-5

    def test_graded_relevance(self):
        # Graded: i1=5, i2=3, i3=1; scores match relevance order
        # top3=[i1,i2,i3]; DCG=5/log2(2)+3/log2(3)+1/log2(4)
        # IDCG=same → NDCG=1.0
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3"],
                "relevance": [5.0, 3.0, 1.0],
                "score": [0.9, 0.8, 0.3],
            }
        )
        result = NDCGAtKMetric().compute(df, None, _schema(), _spec("ndcg_at_k", k=3))
        assert abs(result.value - 1.0) < 1e-5

    def test_no_relevant_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.5],
            }
        )
        result = NDCGAtKMetric().compute(df, None, _schema(), _spec("ndcg_at_k", k=2))
        assert result.value == 0.0

    def test_rank_mode(self):
        schema = RecSysSchema(score_col="rank", recommendations_type="rank")
        result = NDCGAtKMetric().compute(
            _rank_df(), None, schema, _spec("ndcg_at_k", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_k_larger_than_catalogue(self):
        # Only 3 items, k=10 — should not raise, clips to available items
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3"],
                "relevance": [1, 1, 1],
                "score": [0.9, 0.8, 0.7],
            }
        )
        result = NDCGAtKMetric().compute(df, None, _schema(), _spec("ndcg_at_k", k=10))
        assert abs(result.value - 1.0) < 1e-5

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            NDCGAtKMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("ndcg_at_k", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = NDCGAtKMetric().compute(df, None, _schema(), _spec("ndcg_at_k", k=3))
        assert abs(result.value - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# MRRAtKMetric
# ---------------------------------------------------------------------------


class TestMRRAtK:
    """MRR@K tests.

    Fixture (k=3):
      user_1: top3=[i1,i2,i3], relevant={i1,i2,i3} → first hit rank 1 → RR=1.0
      user_2: top3=[i5,i4,i3], relevant={i4,i5}    → first hit rank 1 → RR=1.0
      mean MRR@3 = 1.0
    """

    def test_metric_type(self):
        assert MRRAtKMetric.metric_type == MetricType.recsys

    def test_requires_no_reference(self):
        assert MRRAtKMetric.requires_reference is False

    def test_k3_first_hit_rank1(self):
        result = MRRAtKMetric().compute(
            _interactions_df(), None, _schema(), _spec("mrr_at_k", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_first_hit_at_rank2(self):
        # user_1: top3=[i4,i1,i2], relevant={i1,i2,i3} → first hit rank 2 → RR=0.5
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3", "i4", "i5"],
                "relevance": [1, 1, 1, 0, 0],
                "score": [0.8, 0.7, 0.3, 0.9, 0.1],  # i4 scored highest (not relevant)
            }
        )
        result = MRRAtKMetric().compute(df, None, _schema(), _spec("mrr_at_k", k=3))
        assert abs(result.value - 0.5) < 1e-5

    def test_no_hit_returns_zero(self):
        # top1=[i4] (not relevant), k=1 → RR=0
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i4"],
                "relevance": [1, 0],
                "score": [0.3, 0.9],
            }
        )
        result = MRRAtKMetric().compute(df, None, _schema(), _spec("mrr_at_k", k=1))
        assert result.value == 0.0

    def test_no_relevant_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.5],
            }
        )
        result = MRRAtKMetric().compute(df, None, _schema(), _spec("mrr_at_k", k=2))
        assert result.value == 0.0

    def test_rank_mode(self):
        schema = RecSysSchema(score_col="rank", recommendations_type="rank")
        result = MRRAtKMetric().compute(
            _rank_df(), None, schema, _spec("mrr_at_k", k=3)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            MRRAtKMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("mrr_at_k", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = MRRAtKMetric().compute(df, None, _schema(), _spec("mrr_at_k", k=3))
        assert abs(result.value - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_precision_at_k_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("precision_at_k") is not None

    def test_recall_at_k_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("recall_at_k") is not None

    def test_fbeta_at_k_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("fbeta_at_k") is not None

    def test_hit_rate_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("hit_rate") is not None

    def test_map_at_k_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("map_at_k") is not None

    def test_ndcg_at_k_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("ndcg_at_k") is not None

    def test_mrr_at_k_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("mrr_at_k") is not None

    def test_diversity_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("diversity") is not None

    def test_novelty_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("novelty") is not None

    def test_popularity_bias_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("popularity_bias") is not None

    def test_personalization_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("personalization") is not None

    def test_item_bias_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("item_bias") is not None

    def test_user_bias_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("user_bias") is not None

    def test_serendipity_registered(self):
        from ayn_ml.metrics.registry import get_metric

        assert get_metric("serendipity") is not None


# ---------------------------------------------------------------------------
# SerendipityMetric
# ---------------------------------------------------------------------------


def _serendipity_df() -> pd.DataFrame:
    """Two users, three items with orthogonal feature vectors.

    i1=[1,0], i2=[0,1] are orthogonal (cosine distance = 1).
    i3=[1,1]/√2 is at 45° from both.

    user_1: i1 is relevant (score=0.9), i2 is not (score=0.8)
    user_2: i2 is relevant (score=0.9), i1 is not (score=0.8)
    """
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i1", "i2", "i3"],
            "relevance": [1, 0, 0, 0, 1, 0],
            "score": [0.9, 0.8, 0.3, 0.3, 0.9, 0.5],
            "f1": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "f2": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        }
    )


def _serendipity_train() -> pd.DataFrame:
    """Training: u1 interacted with i2 (opposite of i1), u2 interacted with i1."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i2", "i1"],
            "relevance": [1, 1],
        }
    )


class TestSerendipity:
    def test_metric_type(self):
        assert SerendipityMetric.metric_type == MetricType.recsys

    def test_requires_reference(self):
        assert SerendipityMetric.requires_reference is True

    def test_result_in_unit_interval(self):
        result = SerendipityMetric().compute(
            _serendipity_df(),
            _serendipity_train(),
            _schema(),
            _spec("serendipity", k=2, item_features=["f1", "f2"]),
        )
        assert 0.0 <= result.value <= 1.0

    def test_familiar_item_reduces_serendipity(self):
        # u1's training item is i2; recommending i2 at top is familiar → low serendipity
        # u1's top1=[i1] (relevant, orthogonal to i2 profile) → high serendipity
        # u1's top1=[i2] (not relevant in current) → serendipity=0 (relevance=0)
        result = SerendipityMetric().compute(
            _serendipity_df(),
            _serendipity_train(),
            _schema(),
            _spec("serendipity", k=1, item_features=["f1", "f2"]),
        )
        # top1 for u1=[i1] (relevant, cosine dist from i2 profile = 1.0) → 1.0
        # top1 for u2=[i2] (relevant, cosine dist from i1 profile = 1.0) → 1.0
        assert abs(result.value - 1.0) < 1e-4

    def test_non_relevant_item_contributes_zero(self):
        # All items not relevant → serendipity = 0
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [0, 0],
                "score": [0.9, 0.8],
                "f1": [1.0, 0.0],
                "f2": [0.0, 1.0],
            }
        )
        train = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "relevance": [1]})
        result = SerendipityMetric().compute(
            df, train, _schema(), _spec("serendipity", k=2, item_features=["f1", "f2"])
        )
        assert result.value == 0.0

    def test_reference_none_raises(self):
        with pytest.raises(SchemaError, match="reference"):
            SerendipityMetric().compute(
                _serendipity_df(), None, _schema(),
                _spec("serendipity", k=2, item_features=["f1", "f2"]),
            )

    def test_missing_item_features_raises(self):
        with pytest.raises(SchemaError, match="item_features"):
            SerendipityMetric().compute(
                _serendipity_df(), _serendipity_train(), _schema(),
                _spec("serendipity", k=2),
            )

    def test_missing_feature_column_raises(self):
        with pytest.raises(SchemaError):
            SerendipityMetric().compute(
                _serendipity_df(), _serendipity_train(), _schema(),
                _spec("serendipity", k=2, item_features=["f1", "missing"]),
            )

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            SerendipityMetric().compute(
                _serendipity_df(), _serendipity_train(), TabularSchema(),
                _spec("serendipity", k=2, item_features=["f1", "f2"]),
            )

    def test_unknown_training_user_uses_max_unexpectedness(self):
        # User not in training → profile is None → unexpectedness=1 for relevant items
        train = pd.DataFrame(
            {"user_id": ["u_other"], "item_id": ["i1"], "relevance": [1]}
        )
        result = SerendipityMetric().compute(
            _serendipity_df(), train, _schema(),
            _spec("serendipity", k=1, item_features=["f1", "f2"]),
        )
        # top1 u1=[i1] relevant + no profile → unexpectedness=1 → serendipity=1
        # top1 u2=[i2] relevant + no profile → unexpectedness=1 → serendipity=1
        assert abs(result.value - 1.0) < 1e-4

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_serendipity_df())
        ref = pl.from_pandas(_serendipity_train())
        result = SerendipityMetric().compute(
            df, ref, _schema(),
            _spec("serendipity", k=2, item_features=["f1", "f2"]),
        )
        assert 0.0 <= result.value <= 1.0


# ---------------------------------------------------------------------------
# Fixtures shared by beyond-accuracy tests
# ---------------------------------------------------------------------------


def _features_df() -> pd.DataFrame:
    """Interactions with item feature columns for Diversity tests.

    Items: i1=[1,0], i2=[0,1], i3=[1,1]/√2 — i1⊥i2, i3 at 45°
    user_1 top2 (scores: i1>i2): [i1, i2]  → orthogonal → diversity=1.0
    """
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "relevance": [1, 1, 0],
            "score": [0.9, 0.8, 0.3],
            "f1": [1.0, 0.0, 1.0],
            "f2": [0.0, 1.0, 1.0],
        }
    )


def _training_df() -> pd.DataFrame:
    """Training interactions for Novelty / PopularityBias tests.

    i1: 6/10 = 0.6,  i2: 3/10 = 0.3,  i3: 1/10 = 0.1
    """
    return pd.DataFrame(
        {
            "user_id": ["u0"] * 10,
            "item_id": ["i1", "i1", "i1", "i1", "i1", "i1", "i2", "i2", "i2", "i3"],
            "relevance": [1] * 10,
        }
    )


# ---------------------------------------------------------------------------
# DiversityMetric
# ---------------------------------------------------------------------------


class TestDiversity:
    def test_metric_type(self):
        assert DiversityMetric.metric_type == MetricType.recsys

    def test_orthogonal_items_diversity_one(self):
        # i1=[1,0], i2=[0,1] are orthogonal → cosine distance = 1 → diversity = 1.0
        result = DiversityMetric().compute(
            _features_df(), None, _schema(), _spec("diversity", k=2, item_features=["f1", "f2"])
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_identical_items_diversity_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [1, 1],
                "score": [0.9, 0.8],
                "f1": [1.0, 1.0],
                "f2": [0.0, 0.0],
            }
        )
        result = DiversityMetric().compute(
            df, None, _schema(), _spec("diversity", k=2, item_features=["f1", "f2"])
        )
        assert abs(result.value - 0.0) < 1e-5

    def test_missing_item_features_param_raises(self):
        with pytest.raises(SchemaError, match="item_features"):
            DiversityMetric().compute(
                _features_df(), None, _schema(), _spec("diversity", k=2)
            )

    def test_missing_feature_column_raises(self):
        with pytest.raises(SchemaError):
            DiversityMetric().compute(
                _features_df(), None, _schema(),
                _spec("diversity", k=2, item_features=["f1", "missing_col"]),
            )

    def test_zero_norm_vector_filtered_out(self):
        # i3 has all-zero features → filtered; top2=[i1,i3] but only i1 survives → < 2 valid → 0
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "item_id": ["i1", "i2", "i3"],
                "relevance": [1, 1, 0],
                "score": [0.9, 0.5, 0.3],
                "f1": [1.0, 0.0, 0.0],
                "f2": [0.0, 0.0, 0.0],  # i2 and i3 are zero-norm
            }
        )
        # top2=[i1,i2]; i1 valid, i2 zero-norm → only 1 valid vector → diversity=0
        result = DiversityMetric().compute(
            df, None, _schema(), _spec("diversity", k=2, item_features=["f1", "f2"])
        )
        assert 0.0 <= result.value <= 1.0  # must not exceed [0,1]
        assert result.value == 0.0  # < 2 valid vectors → 0

    def test_single_item_top_k_returns_zero(self):
        # k=1 → only one item per user → no pairs → diversity=0
        result = DiversityMetric().compute(
            _features_df(), None, _schema(), _spec("diversity", k=1, item_features=["f1", "f2"])
        )
        assert result.value == 0.0

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            DiversityMetric().compute(
                _features_df(), None, TabularSchema(),
                _spec("diversity", k=2, item_features=["f1", "f2"]),
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_features_df())
        result = DiversityMetric().compute(
            df, None, _schema(), _spec("diversity", k=2, item_features=["f1", "f2"])
        )
        assert abs(result.value - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# NoveltyMetric
# ---------------------------------------------------------------------------


class TestNovelty:
    def test_metric_type(self):
        assert NoveltyMetric.metric_type == MetricType.recsys

    def test_requires_reference(self):
        assert NoveltyMetric.requires_reference is True

    def test_known_popularity(self):
        # recommend i1 (pop=0.6) and i2 (pop=0.3) to user_1
        # novelty(i1) = -log2(0.6) ≈ 0.737, novelty(i2) = -log2(0.3) ≈ 1.737
        # mean = (0.737 + 1.737) / 2 ≈ 1.237
        import math
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [1, 1],
                "score": [0.9, 0.8],
            }
        )
        result = NoveltyMetric().compute(df, _training_df(), _schema(), _spec("novelty", k=2))
        expected = (-math.log2(0.6) + -math.log2(0.3)) / 2
        assert abs(result.value - expected) < 1e-4

    def test_reference_none_raises(self):
        with pytest.raises(SchemaError, match="reference"):
            NoveltyMetric().compute(
                _interactions_df(), None, _schema(), _spec("novelty", k=3)
            )

    def test_unseen_item_uses_min_popularity(self):
        # i_new not in training → gets min_pop = 1/10 → novelty = log2(10) ≈ 3.32
        df = pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i_new"],
                "relevance": [1],
                "score": [0.9],
            }
        )
        result = NoveltyMetric().compute(df, _training_df(), _schema(), _spec("novelty", k=1))
        import math
        assert abs(result.value - math.log2(10)) < 1e-4

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            NoveltyMetric().compute(
                _interactions_df(), _training_df(), TabularSchema(), _spec("novelty", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        ref = pl.from_pandas(_training_df())
        result = NoveltyMetric().compute(df, ref, _schema(), _spec("novelty", k=3))
        assert result.value > 0.0


# ---------------------------------------------------------------------------
# PopularityBiasMetric
# ---------------------------------------------------------------------------


class TestPopularityBias:
    def test_metric_type(self):
        assert PopularityBiasMetric.metric_type == MetricType.recsys

    def test_requires_reference(self):
        assert PopularityBiasMetric.requires_reference is True

    def test_known_popularity(self):
        # recommend i1 (pop=0.6) and i2 (pop=0.3) → mean = 0.45
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [1, 1],
                "score": [0.9, 0.8],
            }
        )
        result = PopularityBiasMetric().compute(
            df, _training_df(), _schema(), _spec("popularity_bias", k=2)
        )
        assert abs(result.value - 0.45) < 1e-5

    def test_reference_none_raises(self):
        with pytest.raises(SchemaError, match="reference"):
            PopularityBiasMetric().compute(
                _interactions_df(), None, _schema(), _spec("popularity_bias", k=3)
            )

    def test_unseen_item_contributes_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i_new"],
                "relevance": [1],
                "score": [0.9],
            }
        )
        result = PopularityBiasMetric().compute(
            df, _training_df(), _schema(), _spec("popularity_bias", k=1)
        )
        assert result.value == 0.0

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            PopularityBiasMetric().compute(
                _interactions_df(), _training_df(), TabularSchema(), _spec("popularity_bias", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        ref = pl.from_pandas(_training_df())
        result = PopularityBiasMetric().compute(df, ref, _schema(), _spec("popularity_bias", k=3))
        assert 0.0 <= result.value <= 1.0


# ---------------------------------------------------------------------------
# PersonalizationMetric
# ---------------------------------------------------------------------------


class TestPersonalization:
    def test_metric_type(self):
        assert PersonalizationMetric.metric_type == MetricType.recsys

    def test_identical_lists_returns_zero(self):
        # Both users get [i1, i2] → cosine_sim = 1 → personalization = 0
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i1", "i2"],
                "relevance": [1, 1, 1, 1],
                "score": [0.9, 0.8, 0.9, 0.8],
            }
        )
        result = PersonalizationMetric().compute(
            df, None, _schema(), _spec("personalization", k=2)
        )
        assert abs(result.value - 0.0) < 1e-5

    def test_disjoint_lists_returns_one(self):
        # u1 gets [i1, i2], u2 gets [i3, i4] → cosine_sim = 0 → personalization = 1
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i3", "i4"],
                "relevance": [1, 1, 1, 1],
                "score": [0.9, 0.8, 0.9, 0.8],
            }
        )
        result = PersonalizationMetric().compute(
            df, None, _schema(), _spec("personalization", k=2)
        )
        assert abs(result.value - 1.0) < 1e-5

    def test_single_user_returns_zero(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "item_id": ["i1", "i2"],
                "relevance": [1, 1],
                "score": [0.9, 0.8],
            }
        )
        result = PersonalizationMetric().compute(
            df, None, _schema(), _spec("personalization", k=2)
        )
        assert result.value == 0.0

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            PersonalizationMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("personalization", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = PersonalizationMetric().compute(df, None, _schema(), _spec("personalization", k=3))
        assert 0.0 <= result.value <= 1.0


# ---------------------------------------------------------------------------
# ItemBiasMetric
# ---------------------------------------------------------------------------


class TestItemBias:
    def test_metric_type(self):
        assert ItemBiasMetric.metric_type == MetricType.recsys

    def test_equal_frequency_returns_zero(self):
        # u1→[i1], u2→[i2] — each item appears once → Gini=0
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i3", "i4"],
                "relevance": [1, 1, 1, 1],
                "score": [0.9, 0.8, 0.9, 0.8],
            }
        )
        result = ItemBiasMetric().compute(df, None, _schema(), _spec("item_bias", k=1))
        # Each of 4 users gets k=1 → 4 items each appearing once → Gini=0
        # Wait, 2 users here, k=1: u1→[i1], u2→[i3] → counts=[1,1] → Gini=0
        assert abs(result.value - 0.0) < 1e-5

    def test_single_unique_item_returns_zero(self):
        # Both users have i1 as top-1 → only 1 unique item in recommendations → Gini=0
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "item_id": ["i1", "i2", "i1", "i2"],
                "relevance": [1, 0, 1, 0],
                "score": [0.9, 0.1, 0.9, 0.1],
            }
        )
        result = ItemBiasMetric().compute(df, None, _schema(), _spec("item_bias", k=1))
        assert result.value == 0.0

    def test_two_items_unequal_counts(self):
        # 3 users: u1→[i1], u2→[i1], u3→[i2] → counts=[2,1]
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "item_id": ["i1", "i1", "i2"],
                "relevance": [1, 1, 1],
                "score": [0.9, 0.9, 0.9],
            }
        )
        result = ItemBiasMetric().compute(df, None, _schema(), _spec("item_bias", k=1))
        # counts=[1,2] sorted → Gini = (2*(1*1+2*2))/(2*3) - 3/2 = (2*5)/6 - 1.5 = 10/6 - 9/6 = 1/6
        assert abs(result.value - 1 / 6) < 1e-5

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            ItemBiasMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("item_bias", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = ItemBiasMetric().compute(df, None, _schema(), _spec("item_bias", k=3))
        assert 0.0 <= result.value < 1.0


# ---------------------------------------------------------------------------
# UserBiasMetric
# ---------------------------------------------------------------------------


class TestUserBias:
    def test_metric_type(self):
        assert UserBiasMetric.metric_type == MetricType.recsys

    def test_equal_lengths_returns_zero(self):
        # All users get k=3 items (catalogue >= k) → Gini=0
        result = UserBiasMetric().compute(
            _interactions_df(), None, _schema(), _spec("user_bias", k=3)
        )
        assert abs(result.value - 0.0) < 1e-5

    def test_unequal_lengths_nonzero_gini(self):
        # u1 has 3 items, u2 has 1 item → k=3 clips u2 to 1 → lengths=[3,1]
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2"],
                "item_id": ["i1", "i2", "i3", "i4"],
                "relevance": [1, 1, 0, 1],
                "score": [0.9, 0.8, 0.7, 0.9],
            }
        )
        result = UserBiasMetric().compute(df, None, _schema(), _spec("user_bias", k=3))
        # lengths=[3,1] sorted → Gini=(2*(1*1+2*3))/(2*4) - 3/2 = (2*7)/8 - 1.5 = 14/8-12/8=2/8=0.25
        assert abs(result.value - 0.25) < 1e-5

    def test_wrong_schema_raises(self):
        with pytest.raises(SchemaError):
            UserBiasMetric().compute(
                _interactions_df(), None, TabularSchema(), _spec("user_bias", k=3)
            )

    def test_polars_input(self):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(_interactions_df())
        result = UserBiasMetric().compute(df, None, _schema(), _spec("user_bias", k=3))
        assert 0.0 <= result.value < 1.0
