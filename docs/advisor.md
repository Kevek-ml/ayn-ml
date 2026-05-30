# Metric Advisor — `ayn_ml/advisor/`

`MetricAdvisor` analyses your data and builds a ready-to-use `MonitoringPlan` automatically.
Instead of hand-picking drift tests and performance metrics, you describe the task and hand over
a DataFrame — the advisor routes each feature to the statistically appropriate metrics based on
column type, sample size, normality, skewness, and (optionally) variance relative to a reference window.

## Quick start

```python
from ayn_ml.advisor import MetricAdvisor
from ayn_ml.core.schema import TabularSchema

schema = TabularSchema(
    label_col="y_true",
    prediction_col="y_pred",
    proba_col="y_prob",
)
designer = MetricAdvisor(schema)

result = designer.suggest(
    df_current,
    ref=df_reference,   # required — training baseline or historical window
    task_type="classification",
    name="fraud_v2_plan",
)

plan   = result.plan      # MonitoringPlan — pass directly to Runner
warns  = result.warnings  # tuple[str, ...] — why each metric was + / excluded
```

`MetricAdvisor` is reusable — create one instance per schema and call `suggest()` for each
time window with the appropriate cur and ref DataFrames.
## Decision trees

### Drift — per feature column

```
column type = categorical?
  → PSI always
  → chisquare if registered

column type = numeric / binary:
  n_current < 30     → wasserstein only  (+W)
  n_current > 50 000 → PSI + wasserstein only (no hypothesis tests)
    [normality and skewness computed on the REFERENCE distribution]
    is_normal AND |skewness| < 1.0?  → ttest (Welch, equal_var=False)
    + cramervonmises + wasserstein + PSI
    variance_ratio > 1.5 or < 0.67?  → +levene  (+W)
```
> **Why ref for normality?** The ttest / Mann-Whitney U choice should reflect the shape
> of the distribution you are monitoring *against* (the stable baseline), not the
> potentially-drifted cur window. Using the cur window would be circular: drift could
> make it non-normal, silently switching the test. When the ref column is absent,
> normality falls back to the cur window with a debug log.
**Normality test selection** (thresholds from §3.3 of the selection guide):

| Sample size | Test used |
|---|---|
| n < 8 | non-normal (no test — too few points) |
| 8 ≤ n ≤ 300 | Shapiro-Wilk |
| 300 < n ≤ 5 000 | D'Agostino k² (`scipy.stats.normaltest`) |
| n > 5 000 | `\|skewness\| < 1.0` heuristic |

### Performance — classification

| Imbalance ratio | Metrics |
|---|---|
| > 10 : 1 (severe) | f1 + aucpr; accuracy excluded (+ W) |
| > 5 : 1 (moderate) | f1 + auc; accuracy excluded (+ W) |
| ≤ 5 : 1 (balanced) | accuracy |

`accuracy` is excluded when imbalance > 5:1 — it is misleading in those regimes and a warning
is always emitted.

### Performance — regression

`mae + r2`
### `MetricAdvisor(schema)`

| Parameter | Type | Description |
|---|---|---|
| `schema` | `TabularSchema` | Schema shared across all `suggest()` calls |

### `MetricAdvisor.suggest(df, *, ref, task_type, name, model_id, model_version)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | DataFrame | — | Current-window data (pandas or Polars) |
| `ref` | DataFrame | — | Reference window (training baseline). Required. Used for normality routing and variance-ratio / Levene routing |
| `task_type` | str | `"classification"` | `"classification"` \| `"regression"` |
| `name` | str | `"suggested_plan"` | Name embedded in the generated plan |
| `model_id` | str | `""` | Model id embedded in the plan |
| `model_version` | str | `""` | Model version str embedded in the plan |

Returns a `SuggestedPlan`.

### `SuggestedPlan`

```python
@dataclass(frozen=True)
class SuggestedPlan:
    plan: MonitoringPlan        # ready to pass to Runner
    warnings: tuple[str, ...]   # advisory messages explaining routing decisions
```

```python
result.to_dict()
#   "plan": { ... },                        # MonitoringPlan.model_dump()
#   "warnings": ["levene + for ...", ...]
```
## Warnings ref

| Warning | Meaning |
|---|---|
| `"'{col}': only wasserstein suggested — n={n} is too small for hypothesis tests (< 30)"` | Sample size below the min for any test |
| `"levene + for '{col}': variance_ratio={vr:.2f}"` | Variance differs substantially from ref |
| `"accuracy excluded: imbalance ratio {r:.1f}:1 (severe imbalance)"` | Severe class imbalance (> 10:1) |
| `"accuracy demoted: imbalance ratio {r:.1f}:1"` | Moderate class imbalance (> 5:1) |
## End-to-end example

```python
import pandas as pd
from ayn_ml.advisor import MetricAdvisor
from ayn_ml.core.schema import TabularSchema
from ayn_ml.runner import Runner

schema = TabularSchema(
    label_col="y_true",
    prediction_col="y_pred",
    proba_col="y_prob",
)
designer = MetricAdvisor(schema)

# Suggest a plan from a representative sample
result = designer.suggest(
    df_train_sample,
    ref=df_baseline,
    task_type="classification",
    name="fraud_monitor_v2",
    model_id="fraud_model",
    model_version="2.0",
)

for w in result.warnings:
    print("Advisory:", w)

# Inspect the generated specs
for spec in result.plan.metrics:
    feat = f" [{spec.feature_name}]" if spec.feature_name else ""
    print(f"  {spec.name}{feat}")

# Run the plan on prod data
report = Runner().run(
    result.plan,
    cur=df_production,
    ref=df_baseline,
)
print(report.to_dataframe())
```
→ [Notebook: 09 — MetricAdvisor walkthrough](https://github.com/Kevek-ml/ayn-ml/blob/main/examples/09_advisor.ipynb)

---

## Pro Tiers — `ayn-ml-pro`

`ayn-ml-pro` ships two `MetricAdvisor` subclasses that extend the minimal tier from `ayn-ml` (Apache 2.0) with progressively richer automatic metric selection.

```
MetricAdvisor (ayn-ml Apache 2.0)
  └── StandardMetricAdvisor       ← standard tier
        └── ComprehensiveMetricAdvisor  ← comprehensive tier
```

Both classes implement the same `.suggest()` interface as the `ayn-ml` base and return a `(MonitoringPlan, list[str])` result via `result.plan` and `result.warnings`.

## StandardMetricAdvisor

**Import:** `from ayn_ml_pro.advisor import StandardMetricAdvisor`

Adds richer performance metrics and descriptive stats on top of the minimal tier. Drift routing is identical to the Apache 2.0 minimal tier.

### Additions over the minimal tier

| Category | Added metrics |
|---|---|
| Regression | `mse`, `mape` (alongside `mae` + `r2`) |
| Classification (balanced) | `f1`, `auc` (alongside `accuracy`) |
| Numeric columns | `mean`, `std` |
| Categorical columns | `top_category` |

```python
from ayn_ml.core.schema import TabularSchema
from ayn_ml_pro.advisor import StandardMetricAdvisor

schema = TabularSchema(label_col="y_true", prediction_col="y_pred", proba_col="y_prob")
designer = StandardMetricAdvisor(schema)
result = designer.suggest(df_current, ref=df_reference, task_type="classification", name="fraud_monitor_standard")

plan  = result.plan
warns = result.warnings
```

## ComprehensiveMetricAdvisor

**Import:** `from ayn_ml_pro.advisor import ComprehensiveMetricAdvisor`

Adds MMD drift detection, the full performance suite, and extended distributional stats on top of the standard tier.

### Additions over the standard tier

| Category | Added metrics | Condition |
|---|---|---|
| Drift | `mmd` (Maximum Mean Discrepancy) | Numeric columns with n >= 200 |
| Classification | `precision`, `recall` | All imbalance regimes |
| Classification (balanced) | `log_loss`, `brier` | Balanced datasets only |
| Numeric / binary columns | `skewness`, `kurtosis` | All numeric and binary columns |

```python
from ayn_ml.core.schema import TabularSchema
from ayn_ml_pro.advisor import ComprehensiveMetricAdvisor

schema = TabularSchema(label_col="y_true", prediction_col="y_pred", proba_col="y_prob")
designer = ComprehensiveMetricAdvisor(schema)
result = designer.suggest(df_current, ref=df_reference, task_type="classification", name="fraud_monitor_comprehensive")
warns = result.warnings
```

## Choosing a tier

| Situation | Recommended tier |
|---|---|
| Fast feedback, limited compute | `StandardMetricAdvisor` |
| Deep statistical monitoring, model audit | `ComprehensiveMetricAdvisor` |
| Custom metric selection | Use `MetricAdvisor` from `ayn-ml` directly and supply your own `MetricSpec` list |

## Tier comparison at a glance

| Metric | Minimal (ayn-ml) | Standard | Comprehensive |
|---|---|---|---|
| KS test (numeric drift) | ✓ | ✓ | ✓ |
| Chi-squared (categorical drift) | ✓ | ✓ | ✓ |
| MMD (numeric drift, n>=200) | | | ✓ |
| `mae`, `r2` (regression) | ✓ | ✓ | ✓ |
| `mse`, `mape` (regression) | | ✓ | ✓ |
| `accuracy` (classification) | ✓ | ✓ | ✓ |
| `f1`, `auc` (balanced) | | ✓ | ✓ |
| `precision`, `recall` | | | ✓ |
| `log_loss`, `brier` (balanced) | | | ✓ |
| `mean`, `std` (numeric) | | ✓ | ✓ |
| `top_category` (categorical) | | ✓ | ✓ |
| `skewness`, `kurtosis` | | | ✓ |
