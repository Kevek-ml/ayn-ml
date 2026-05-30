import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ayn_ml.core.schema import AgentSchema, RecSysSchema, TabularSchema, TextSchema
from ayn_ml.core.spec import MonitoringPlan


class TestTabularSchema:
    def test_defaults(self):
        s = TabularSchema()
        assert s.label_col == "y_true"
        assert s.prediction_col == "y_pred"
        assert s.proba_col == "y_pred_proba"
        assert s.timestamp_col is None
        assert s.feature_types == {}
        assert s.model_id_col is None
        assert s.model_version_col is None

    def test_no_feature_cols_field(self):
        s = TabularSchema()
        assert not hasattr(s, "feature_cols")

    def test_custom_columns(self):
        s = TabularSchema(label_col="label", prediction_col="score", proba_col=None)
        assert s.label_col == "label"
        assert s.proba_col is None

    def test_model_identity_cols_can_be_none(self):
        s = TabularSchema(model_id_col=None, model_version_col=None)
        assert s.model_id_col is None
        assert s.model_version_col is None

    def test_model_identity_cols_custom_names(self):
        s = TabularSchema(model_id_col="mid", model_version_col="mver")
        assert s.model_id_col == "mid"
        assert s.model_version_col == "mver"

    def test_frozen(self):
        s = TabularSchema()
        with pytest.raises(ValidationError):
            s.label_col = "other"

    def test_type_discriminator(self):
        assert TabularSchema().type == "tabular"

    def test_explicit_feature_types(self):
        s = TabularSchema(feature_types={"age": "numeric", "region": "categorical"})
        assert s.feature_types["age"] == "numeric"
        assert s.feature_types["region"] == "categorical"


class TestTabularSchemaColumnNames:
    def test_includes_mandatory_cols(self):
        s = TabularSchema()
        cols = s.column_names
        assert "y_true" in cols
        assert "y_pred" in cols
        assert "timestamp" not in cols  # timestamp_col defaults to None

    def test_includes_timestamp_when_configured(self):
        s = TabularSchema(timestamp_col="ts")
        assert "ts" in s.column_names

    def test_includes_proba_when_set(self):
        s = TabularSchema()
        assert "y_pred_proba" in s.column_names

    def test_excludes_proba_when_none(self):
        s = TabularSchema(proba_col=None)
        assert "y_pred_proba" not in s.column_names

    def test_excludes_optional_identity_cols_when_none(self):
        s = TabularSchema(model_id_col=None, model_version_col=None)
        assert "model_id" not in s.column_names
        assert "model_version" not in s.column_names

    def test_no_duplicates(self):
        cols = TabularSchema().column_names
        assert len(cols) == len(set(cols))


class TestTabularSchemaFromDataframe:
    def _df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "age": rng.integers(20, 60, 100),
                "income": rng.normal(50_000, 10_000, 100),
                "region": np.zeros(100, dtype=int),
                "category": rng.choice(["A", "B", "C"], 100),
            }
        )

    def test_schema_cols_excluded_from_feature_types(self):
        df = self._df().copy()
        df["y_true"] = 0
        df["y_pred"] = 0
        df["y_pred_proba"] = 0.0
        df["ts"] = "2024-01-01"
        s = TabularSchema.from_dataframe(df, timestamp_col="ts")
        for col in ("y_true", "y_pred", "y_pred_proba", "ts"):
            assert col not in s.feature_types

    def test_infers_numeric_for_float(self):
        s = TabularSchema.from_dataframe(self._df())
        assert s.feature_types["income"] == "numeric"

    def test_infers_categorical_for_object(self):
        s = TabularSchema.from_dataframe(self._df())
        assert s.feature_types["category"] == "categorical"

    def test_override_corrects_integer_encoded_categorical(self):
        s = TabularSchema.from_dataframe(self._df(), feature_types={"region": "categorical"})
        assert s.feature_types["region"] == "categorical"

    def test_inference_without_override_wrong_for_int_encoded(self):
        s = TabularSchema.from_dataframe(self._df())
        assert s.feature_types["region"] == "numeric"

    def test_kwargs_forwarded(self):
        s = TabularSchema.from_dataframe(self._df(), label_col="target")
        assert s.label_col == "target"

    def test_schema_is_frozen(self):
        s = TabularSchema.from_dataframe(self._df())
        with pytest.raises(ValidationError):
            s.label_col = "other"

    def test_infers_from_polars_dataframe(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"age": [20, 30, 40], "category": ["A", "B", "C"]})
        s = TabularSchema.from_dataframe(df)
        assert s.feature_types["age"] == "numeric"
        assert s.feature_types["category"] == "categorical"


class TestTextSchema:
    def test_defaults(self):
        s = TextSchema()
        assert s.input_col == "input_text"
        assert s.output_col == "output_text"
        assert s.reference_col == "reference_text"
        assert s.embedding_col is None

    def test_unsupervised_mode(self):
        s = TextSchema(reference_col=None)
        assert s.reference_col is None

    def test_type_discriminator(self):
        assert TextSchema().type == "text"

    def test_column_names_excludes_none_cols(self):
        s = TextSchema(reference_col=None, embedding_col=None)
        cols = s.column_names
        assert "input_text" in cols
        assert "output_text" in cols
        assert "reference_text" not in cols

    def test_column_names_includes_embedding_when_set(self):
        s = TextSchema(embedding_col="emb")
        assert "emb" in s.column_names


class TestAgentSchema:
    def test_defaults(self):
        s = AgentSchema()
        assert s.trace_col == "trace"
        assert s.tokens_used_col == "tokens_used"
        assert s.cost_col == "cost_usd"

    def test_type_discriminator(self):
        assert AgentSchema().type == "agent"

    def test_column_names_includes_optional_when_set(self):
        s = AgentSchema()
        cols = s.column_names
        assert "success" in cols
        assert "tokens_used" in cols

    def test_column_names_excludes_none_optional_cols(self):
        s = AgentSchema(success_col=None, tool_calls_col=None, tokens_used_col=None, latency_col=None, cost_col=None)
        cols = s.column_names
        assert "success" not in cols
        assert "tokens_used" not in cols


class TestRecSysSchema:
    def test_defaults(self):
        s = RecSysSchema()
        assert s.user_id_col == "user_id"
        assert s.item_id_col == "item_id"
        assert s.relevance_col == "relevance"
        assert s.score_col == "score"
        assert s.recommendations_type == "score"

    def test_type_discriminator(self):
        assert RecSysSchema().type == "recsys"

    def test_frozen(self):
        s = RecSysSchema()
        with pytest.raises(ValidationError):
            s.user_id_col = "other"

    def test_custom_columns(self):
        s = RecSysSchema(
            user_id_col="uid",
            item_id_col="iid",
            relevance_col="rating",
            score_col="predicted",
        )
        assert s.user_id_col == "uid"
        assert s.item_id_col == "iid"
        assert s.relevance_col == "rating"
        assert s.score_col == "predicted"

    def test_score_col_can_be_none(self):
        s = RecSysSchema(score_col=None)
        assert s.score_col is None

    def test_recommendations_type_rank(self):
        s = RecSysSchema(recommendations_type="rank")
        assert s.recommendations_type == "rank"

    def test_invalid_recommendations_type_raises(self):
        with pytest.raises(ValidationError):
            RecSysSchema(recommendations_type="invalid")

    def test_column_names_includes_mandatory_cols(self):
        s = RecSysSchema()
        cols = s.column_names
        assert "user_id" in cols
        assert "item_id" in cols
        assert "relevance" in cols
        assert "score" in cols

    def test_column_names_excludes_score_when_none(self):
        s = RecSysSchema(score_col=None)
        assert "score" not in s.column_names

    def test_column_names_includes_timestamp_when_set(self):
        s = RecSysSchema(timestamp_col="ts")
        assert "ts" in s.column_names

    def test_column_names_no_duplicates(self):
        cols = RecSysSchema().column_names
        assert len(cols) == len(set(cols))


class TestDiscriminatedUnion:
    def test_monitoring_plan_accepts_tabular(self):
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema={"type": "tabular"},
            metrics=[],
        )
        assert isinstance(plan.data_schema, TabularSchema)

    def test_monitoring_plan_accepts_text(self):
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema={"type": "text"},
            metrics=[],
        )
        assert isinstance(plan.data_schema, TextSchema)

    def test_monitoring_plan_accepts_agent(self):
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema={"type": "agent"},
            metrics=[],
        )
        assert isinstance(plan.data_schema, AgentSchema)

    def test_monitoring_plan_accepts_recsys(self):
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema={"type": "recsys"},
            metrics=[],
        )
        assert isinstance(plan.data_schema, RecSysSchema)

    def test_recsys_serialization_roundtrip(self):
        plan = MonitoringPlan(
            name="p",
            model_id="m",
            model_version="1",
            data_schema=RecSysSchema(user_id_col="uid", score_col=None),
            metrics=[],
        )
        data = plan.model_dump()
        plan2 = MonitoringPlan.model_validate(data)
        assert isinstance(plan2.data_schema, RecSysSchema)
        assert plan2.data_schema.user_id_col == "uid"
        assert plan2.data_schema.score_col is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            MonitoringPlan(
                name="p",
                model_id="m",
                model_version="1",
                data_schema={"type": "unknown"},
                metrics=[],
            )
