# Data schemas

A schema maps logical column roles to physical column names in your DataFrame. ayn-ml supports four modalities — pick the one that matches your model type.

## TabularSchema

For supervised ML models (classification and regression).

| Field | Default | Description |
|---|---|---|
| `label_col` | `"y_true"` | Ground-truth labels |
| `prediction_col` | `"y_pred"` | Model predictions |
| `proba_col` | `"y_pred_proba"` | Predicted probabilities (set to `None` to disable) |
| `feature_types` | `{}` | Maps column names to `"numeric"` or `"categorical"` — required to correctly handle integer-encoded categoricals |
| `timestamp_col` | `None` | Observation timestamp. When set, the Runner derives `period_start`/`period_end` from min/max values. Required when `plan.window.type == "time_window"`. |
| `model_id_col` | `None` | Column identifying the model per row. When set, the Runner filters rows to `MonitoringPlan.model_id`. `None` skips filtering. |
| `model_version_col` | `None` | Column identifying the model version per row. Same semantics as `model_id_col`. |

> **Declare only what exists in your DataFrame.** All three optional columns default to `None`. The Runner validates at runtime that every configured column is present and raises `SchemaError` if it is missing (configurable via `Runner(strict=False)`).

**Infer types from a DataFrame:**

```python
schema = TabularSchema.from_dataframe(
    df_reference,
    feature_types={"region": "categorical"},  # correct int-encoded categoricals
    label_col="y_true",
    prediction_col="y_pred",
    model_id_col=None,        # not in the data — falls back to MonitoringPlan.model_id
    model_version_col=None,
)
```

Schema columns (`label_col`, `prediction_col`, etc.) are automatically excluded from type inference — only feature columns appear in `feature_types`.

## TextSchema

For NLP / LLM text-in / text-out models.

| Field | Default | Description |
|---|---|---|
| `input_col` | `"input_text"` | Model input text |
| `output_col` | `"output_text"` | Model output text |
| `reference_col` | `"reference_text"` | Expected output for quality metrics (ROUGE, BLEU); set to `None` for unsupervised mode |
| `embedding_col` | `None` | Pre-computed embeddings for semantic similarity metrics |

## AgentSchema

For AI agent traces (tool-calling, multi-step reasoning).

| Field | Default | Description |
|---|---|---|
| `trace_col` | `"trace"` | Full execution trace |
| `success_col` | `"success"` | Boolean task-completion flag |
| `tool_calls_col` | `"tool_calls"` | Serialized tool-call records |
| `tokens_used_col` | `"tokens_used"` | Total token consumption per run |
| `latency_col` | `"latency_ms"` | End-to-end latency in milliseconds |
| `cost_col` | `"cost_usd"` | Cost in USD per run |

All optional fields default to `None` when not present in the data.

## RecSysSchema

For recommender-system evaluation data. Rows represent individual (user, item) interactions.

| Field | Default | Description |
|---|---|---|
| `user_id_col` | `"user_id"` | Column containing the user identifier |
| `item_id_col` | `"item_id"` | Column containing the item identifier |
| `relevance_col` | `"relevance"` | Ground-truth relevance signal — binary (0/1) or graded |
| `score_col` | `"score"` | Model's predicted score or rank position; set to `None` when the DataFrame is already sorted |
| `recommendations_type` | `"score"` | Whether `score_col` holds raw scores (`"score"`) or explicit rank positions (`"rank"`) |
