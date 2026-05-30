"""Data schema definitions for the four supported modalities.

Each schema is a frozen Pydantic model that maps logical column roles to
physical column names in the user's DataFrame.  The DataSchema union type
supports discriminated deserialization via the ``type`` literal field.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ColumnType(str, Enum):
    """Resolved classification of a DataFrame column used for metric routing.

    Resolution order (highest to lowest priority):

    1. ``TabularSchema.feature_types`` explicit declaration.
    2. numpy dtype inference — correct for float/string columns; may be wrong
       for integer-encoded categoricals, which is why explicit declaration
       exists.

    Values:
        numeric: Continuous or ordinal numeric column (float or multi-valued int).
        categorical: String or integer-encoded categorical column.
        binary: Integer column whose unique values are a subset of ``{0, 1}``.
            Accepted by both numeric tests (KS, Wasserstein) and the chi-square
            homogeneity test.  Inference-only — cannot be declared via
            ``schema.feature_types`` (only ``"numeric"`` and ``"categorical"``
            are valid there).
    """

    numeric = "numeric"
    categorical = "categorical"
    binary = "binary"


class BaseSchema(BaseModel):
    """Shared column names present in every schema.

    All fields default to ``None`` — declare only the columns that exist in
    your DataFrame.  The Runner validates at runtime that every configured
    column is present.

    Attributes:
        timestamp_col: Column containing the observation timestamp.  When set,
            the Runner derives ``period_start`` / ``period_end`` from its
            min/max values and validates the column is present.  Required when
            ``plan.window.type == "time_window"``.
        model_id_col: Column identifying the model that produced the row.
            When set, the Runner filters rows to ``MonitoringPlan.model_id``.
            ``None`` skips model-id filtering.
        model_version_col: Column identifying the model version.  Same
            semantics as ``model_id_col``.
    """

    model_config = ConfigDict(frozen=True)

    timestamp_col: str | None = None
    model_id_col: str | None = None
    model_version_col: str | None = None

    @property
    def column_names(self) -> list[str]:
        """Return all non-None column name values declared by this schema.

        Used by ``required_columns`` to compute the minimal set of columns
        the runner needs to load from the data source.

        Returns:
            Ordered list of column name strings; optional fields set to
            ``None`` are excluded.
        """
        cols = []
        for c in (self.timestamp_col, self.model_id_col, self.model_version_col):
            if c:
                cols.append(c)
        return cols


class TabularSchema(BaseSchema):
    """Schema for supervised tabular ML models (classification and regression).

    Attributes:
        type: Discriminator literal, always ``"tabular"``.
        label_col: Column containing ground-truth labels (y_true).
        prediction_col: Column containing model predictions (y_pred).
        proba_col: Column containing predicted probabilities; required for
            AUC, log-loss, and Brier score.  Set to ``None`` to disable.
        feature_types: Maps column names to ``"numeric"`` or ``"categorical"``.
            Used by drift metrics to determine the appropriate computation path.
            Columns absent from this dict fall back to numpy dtype inference,
            which is correct for float/string columns but wrong for
            integer-encoded categoricals.  Use ``from_dataframe`` to populate
            this automatically with optional overrides.
        protected_cols: Columns representing sensitive or protected attributes
            (e.g. ``["gender", "age_group"]``).  Fairness metrics read
            ``spec.feature_name`` and require it to be listed here when this
            field is set.  ``None`` disables the declaration check (the column
            must still exist in the DataFrame).
    """

    type: Literal["tabular"] = "tabular"

    label_col: str = "y_true"
    prediction_col: str = "y_pred"
    proba_col: str | None = "y_pred_proba"
    feature_types: dict[str, Literal["numeric", "categorical"]] = {}
    protected_cols: list[str] | None = None

    @property
    def column_names(self) -> list[str]:
        """Return all non-None column name values declared by this schema."""
        cols = super().column_names + [self.label_col, self.prediction_col]
        if self.proba_col:
            cols.append(self.proba_col)
        if self.protected_cols:
            cols.extend(self.protected_cols)
        return cols

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        feature_types: dict[str, Literal["numeric", "categorical"]] | None = None,
        **kwargs: Any,
    ) -> TabularSchema:
        """Build a TabularSchema by inferring column types from a DataFrame.

        Only feature columns are inspected — schema columns (``label_col``,
        ``prediction_col``, ``proba_col``, ``timestamp_col``, ``model_id_col``,
        ``model_version_col``) are excluded so they never appear in
        ``feature_types`` and cannot accidentally be fed to drift metrics.

        Remaining columns are classified by numpy dtype:

        - Numeric dtypes (int, float) → ``"numeric"``
        - All other dtypes (object, string, category) → ``"categorical"``

        The optional ``feature_types`` argument overrides inferred values for
        specific columns.  Use it to correct integer-encoded categoricals that
        dtype inference cannot distinguish from genuine numeric features.

        Args:
            df: Source DataFrame (eager pandas or Polars DataFrame).
                Polars ``LazyFrame`` is not supported; call ``.collect()`` first.
            feature_types: Explicit type overrides applied on top of inference.
                Only columns that would be inferred incorrectly need to be
                listed here (e.g. ``{"region": "categorical"}``).
            **kwargs: Additional ``TabularSchema`` fields (``label_col``,
                ``prediction_col``, ``proba_col``, etc.).

        Returns:
            TabularSchema with ``feature_types`` populated from inference
            merged with any explicit overrides.
        """
        import narwhals as nw
        import numpy as np

        # Build a temporary instance to resolve default + kwarg column names.
        tmp = cls(**kwargs)
        exclude = {
            tmp.label_col,
            tmp.prediction_col,
            *(([tmp.timestamp_col]) if tmp.timestamp_col else []),
            *(([tmp.proba_col]) if tmp.proba_col else []),
            *(([tmp.model_id_col]) if tmp.model_id_col else []),
            *(([tmp.model_version_col]) if tmp.model_version_col else []),
            *(tmp.protected_cols if tmp.protected_cols else []),
        }

        native = nw.from_native(df, eager_only=True)
        inferred: dict[str, Literal["numeric", "categorical"]] = {
            col: "numeric" if np.issubdtype(native[col].to_numpy().dtype, np.number) else "categorical"
            for col in native.columns
            if col not in exclude
        }
        return cls(feature_types={**inferred, **(feature_types or {})}, **kwargs)


class TextSchema(BaseSchema):
    """Schema for NLP / LLM text-in / text-out models.

    Attributes:
        type: Discriminator literal, always ``"text"``.
        input_col: Column containing the model input text.
        output_col: Column containing the model output text.
        reference_col: Column containing reference / expected output for
            quality metrics such as ROUGE and BLEU.
        embedding_col: Column containing pre-computed embeddings; used by
            semantic similarity and embedding-drift metrics.
    """

    type: Literal["text"] = "text"

    input_col: str = "input_text"
    output_col: str = "output_text"
    reference_col: str | None = "reference_text"
    embedding_col: str | None = None

    @property
    def column_names(self) -> list[str]:
        """Return all non-None column name values declared by this schema."""
        cols = super().column_names + [self.input_col, self.output_col]
        if self.reference_col:
            cols.append(self.reference_col)
        if self.embedding_col:
            cols.append(self.embedding_col)
        return cols


class AgentSchema(BaseSchema):
    """Schema for AI agent traces (tool-calling, multi-step reasoning).

    Attributes:
        type: Discriminator literal, always ``"agent"``.
        input_col: Column containing the user instruction / prompt.
        output_col: Column containing the agent's final response.
        trace_col: Column containing the full execution trace (tool calls,
            intermediate steps, etc.).
        success_col: Optional column with a boolean task-completion flag.
        tool_calls_col: Optional column with serialized tool-call records.
        tokens_used_col: Optional column with total token consumption per run.
        latency_col: Optional column with end-to-end latency in milliseconds.
        cost_col: Optional column with cost in USD per run.
    """

    type: Literal["agent"] = "agent"

    input_col: str = "input"
    output_col: str = "output"
    trace_col: str = "trace"
    success_col: str | None = "success"
    tool_calls_col: str | None = "tool_calls"
    tokens_used_col: str | None = "tokens_used"
    latency_col: str | None = "latency_ms"
    cost_col: str | None = "cost_usd"

    @property
    def column_names(self) -> list[str]:
        """Return all non-None column name values declared by this schema."""
        cols = super().column_names + [self.input_col, self.output_col, self.trace_col]
        for c in (self.success_col, self.tool_calls_col, self.tokens_used_col, self.latency_col, self.cost_col):
            if c:
                cols.append(c)
        return cols


class RecSysSchema(BaseSchema):
    """Schema for recommender-system evaluation data.

    Rows represent individual (user, item) interactions.  Each row must carry
    a ground-truth relevance signal and, optionally, the predicted score or
    rank produced by the model.

    Attributes:
        type: Discriminator literal, always ``"recsys"``.
        user_id_col: Column containing the user identifier.
        item_id_col: Column containing the item identifier.
        relevance_col: Column containing the ground-truth relevance signal.
            May be binary (0/1) or graded (integer or float).
        score_col: Column containing the model's predicted score (when
            ``recommendations_type="score"``) or predicted rank position
            (when ``recommendations_type="rank"``).  Set to ``None`` when
            the DataFrame is already sorted in descending relevance order
            and no explicit score column is available.
        recommendations_type: Whether ``score_col`` holds raw prediction
            scores (``"score"``, default) or explicit rank positions
            (``"rank"``).  Ignored when ``score_col`` is ``None``.
    """

    type: Literal["recsys"] = "recsys"

    user_id_col: str = "user_id"
    item_id_col: str = "item_id"
    relevance_col: str = "relevance"
    score_col: str | None = "score"
    recommendations_type: Literal["score", "rank"] = "score"

    @property
    def column_names(self) -> list[str]:
        """Return all non-None column name values declared by this schema."""
        cols = super().column_names + [self.user_id_col, self.item_id_col, self.relevance_col]
        if self.score_col:
            cols.append(self.score_col)
        return cols


DataSchema = Annotated[
    TabularSchema | TextSchema | AgentSchema | RecSysSchema,
    Field(discriminator="type"),
]
"""Union of all supported schema types.

Use this annotation on model fields that must accept any modality.  Pydantic
uses the ``type`` discriminator field to deserialize the correct subclass.
"""
