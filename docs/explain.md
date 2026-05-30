# Explain — SHAP Monitoring and LLM Drift Explanation

> **Status:** API and class structure are defined. `ShapDriftMonitor.compare()` and `LlmDriftExplainer.explain()` are not yet implemented.

---

## ShapDriftMonitor

**Import:** `from ayn_ml_pro.explain import ShapDriftMonitor`

**Extra required:** `pip install "ayn-ml-pro[shap]"`

`ShapDriftMonitor` computes SHAP feature-importance values for both the current and reference windows, then applies drift tests (Wasserstein distance, PSI) to each feature's SHAP distribution to rank which features have shifted most.

This answers a different question than standard feature drift: not "has the input distribution of feature X changed?" but "has the model's *reliance* on feature X changed?". A feature whose values are stable can still show SHAP drift if the model started weighting it differently.

### Constructor

```python
ShapDriftMonitor(
    model: Any,                  # Fitted model — sklearn, XGBoost, LightGBM, PyTorch
    feature_names: list[str],    # Column names of the feature matrix (must match model input order)
    max_samples: int = 500,      # Max samples per window used for SHAP computation
)
```

`max_samples` caps computation cost. For large windows, the monitor samples randomly before computing SHAP values.

### Usage

```python
from ayn_ml_pro.explain import ShapDriftMonitor

monitor = ShapDriftMonitor(
    model=fitted_clf,
    feature_names=["age", "income", "score", "region"],
    max_samples=500,
)

report = monitor.compare(df_current, ref=df_reference)

print(report.top_drifting_features)  # ranked list of features by SHAP drift score
```

### Report structure

The SHAP drift report exposes per-feature drift scores and a ranked list of the most drifted features:

```python
report.top_drifting_features    # list[str] — features ranked by drift magnitude
report.feature_scores           # dict[str, float] — drift score per feature
report.method                   # str — drift test used (e.g. "wasserstein")
```

### Choosing `max_samples`

| Dataset size | Recommended `max_samples` |
|---|---|
| < 1 000 rows | Use all rows (set to dataset size) |
| 1 000 – 10 000 | 500 (default) |
| > 10 000 | 300 – 500 |

Tree-based explainers (XGBoost, LightGBM, sklearn tree ensembles) are significantly faster than kernel SHAP — for those models you can safely increase `max_samples`.

---

## LlmDriftExplainer

**Import:** `from ayn_ml_pro.explain import LlmDriftExplainer`

**Extra required:** `pip install "ayn-ml-pro[llm-openai]"` or `pip install "ayn-ml-pro[llm-anthropic]"`

`LlmDriftExplainer` takes a `MonitoringReport` containing detected drift signals and produces a concise natural-language explanation of what drifted, by how much, and what the likely operational impact is.

### Constructor

```python
LlmDriftExplainer(
    provider: str,                           # "openai" or "anthropic"
    model: str,                              # Model name — "gpt-4o", "claude-opus-4", etc.
    api_key: str | None = None,              # Falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY
    langfuse_secret_key: str | None = None,  # Optional Langfuse tracing
    langfuse_public_key: str | None = None,
)
```

API keys must not be hardcoded. Use environment variables:

```python
import os
from ayn_ml_pro.explain import LlmDriftExplainer

explainer = LlmDriftExplainer(
    provider="anthropic",
    model="claude-opus-4-7",
    # api_key falls back to os.environ["ANTHROPIC_API_KEY"]
)
```

### Usage

```python
# Run monitoring first to get a report
report = runner.run_once(df_current, ref=df_reference)

# Generate natural-language explanation
explanation = explainer.explain(report)

print(explanation.summary)           # one-paragraph overview of detected drift
print(explanation.recommendations)  # list[str] — suggested remediation steps
```

### Tracing with Langfuse

Pass Langfuse credentials to log every LLM call for observability and cost tracking:

```python
explainer = LlmDriftExplainer(
    provider="openai",
    model="gpt-4o",
    langfuse_secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    langfuse_public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
)
```

### Supported providers

| Provider | Extra | Default env var |
|----------|-------|-----------------|
| `"openai"` | `llm-openai` | `OPENAI_API_KEY` |
| `"anthropic"` | `llm-anthropic` | `ANTHROPIC_API_KEY` |

---

## Combining SHAP and LLM explanation

A typical workflow runs SHAP monitoring first to rank drifted features, then feeds the standard drift report to the LLM explainer for a human-readable summary:

```python
from ayn_ml_pro.explain import ShapDriftMonitor, LlmDriftExplainer

# Step 1 — rank drifted features by SHAP importance shift
shap_monitor = ShapDriftMonitor(model=clf, feature_names=feature_cols)
shap_report = shap_monitor.compare(df_current, ref=df_reference)
print("Top drifting features:", shap_report.top_drifting_features)

# Step 2 — explain the monitoring report in natural language
explainer = LlmDriftExplainer(provider="anthropic", model="claude-opus-4-7")
monitoring_report = runner.run_once(df_current, ref=df_reference)
explanation = explainer.explain(monitoring_report)
print(explanation.summary)
```
