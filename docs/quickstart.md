# Quick Start

This page walks you through the three main entry points to ayn-ml:
building a plan manually, loading one from YAML, and generating one automatically
with `MetricAdvisor`.

---

## Installation

```bash
pip install ayn-ml
```

For production workloads with Polars:

```bash
pip install ayn-ml[polars]
```

---

## 1 — Define a monitoring plan

A `MonitoringPlan` binds a data schema, model identity, and the list of metrics to
compute. Plans are immutable Pydantic models that round-trip through YAML.

```python
from ayn_ml import MonitoringPlan, TabularSchema, MetricSpec

schema = TabularSchema(
    label_col="y_true",
    prediction_col="y_pred",
    proba_col="y_prob",
    feature_types={"tenure": "categorical"},   # override int-encoded columns
)

plan = MonitoringPlan(
    name="churn_model_monitoring",
    model_id="churn_model",
    model_version="2.1",
    data_schema=schema,
    metrics=[
        MetricSpec(name="accuracy", threshold=0.85, upper_bound=False),
        MetricSpec(name="psi", feature_name="age", threshold=0.2),
        MetricSpec(name="f1", threshold=0.80, upper_bound=False),
    ],
)
```

---

## 2 — Declarative YAML config (roadmap)

`MonitoringPlan` is a Pydantic model — YAML round-trip (`from_yaml` / `to_yaml`) is on the
roadmap. The schema below shows the planned config format for reference:

```yaml
# churn_plan.yaml — planned declarative config format
name: churn_model_monitoring
model_id: churn_model
model_version: "2.1"
data_schema:
  type: tabular
  label_col: y_true
  prediction_col: y_pred
metrics:
  - name: accuracy
    threshold: 0.85
    upper_bound: false
  - name: psi
    feature_name: age
    threshold: 0.2
```

Until YAML I/O ships, build the plan in Python as shown in step 1 above.
`MonitoringPlan` is a standard Pydantic model, so `plan.model_dump()` and
`MonitoringPlan.model_validate(d)` work today for dict-based serialisation.

---

## 3 — Run and inspect results

```python
from ayn_ml.runner import Runner
from ayn_ml.stores import InMemoryStore

store  = InMemoryStore()
report = Runner().run(plan, current=df_current, reference=df_reference, store=store)

print(report.to_dataframe())
#    metric_name  metric_type  value   status  threshold
# 0     accuracy  performance   0.82    False       0.85
# 1          psi        drift   0.31    False       0.20
# 2           f1  performance   0.79    False       0.80
```

---

## 4 — Let MetricAdvisor build the plan

Don't know which drift test to use for a skewed feature, or whether F1 or accuracy is right for
an imbalanced dataset? `MetricAdvisor` answers both from the data itself.

```python
from ayn_ml.advisor import MetricAdvisor
from ayn_ml.core.schema import TabularSchema

schema   = TabularSchema(label_col="y_true", prediction_col="y_pred", proba_col="y_prob")
designer = MetricAdvisor(schema)

result = designer.suggest(
    df_current,
    reference=df_reference,   # required — training baseline or historical window
    task_type="classification",
    name="fraud_monitor",
    model_id="fraud_v2",
    model_version="1.0",
)

plan  = result.plan      # MonitoringPlan — pass directly to Runner
for w in result.warnings:
    print("Advisory:", w)
# Advisory: accuracy demoted: imbalance ratio 7.3:1
# Advisory: levene added for 'income': variance_ratio=1.91
```

The advisor inspects every feature column independently:

- **Numeric** → normality test (Shapiro-Wilk / D'Agostino / skewness heuristic) → t-test or
  Mann-Whitney U + CvM + Wasserstein + PSI; Levene added when variance shifts
- **Categorical** → PSI + chi-square
- **Performance** → accuracy demoted or excluded when class imbalance is detected; F1 + AUCPR promoted
- **Sample-size guards** → Wasserstein only for n < 30; PSI + Wasserstein only for n > 50 000

→ [Full advisor reference](advisor.md)

---

## 5 — Works with pandas and Polars

```python
import pandas as pd
import polars as pl
from ayn_ml.runner import Runner

# Both work — narwhals handles the backend
report_pd = Runner().run(plan, current=df_pandas,  reference=df_ref_pandas)
report_pl = Runner().run(plan, current=df_polars,  reference=df_ref_polars)
```

---

## 6 — Persist results

```python
from ayn_ml.stores import SqliteStore

store  = SqliteStore("monitoring.db")
report = Runner().run(plan, current=df_current, reference=df_reference, store=store)

# Read back the history as flat rows
rows = store.read_history("churn_model", limit=30)
```

For cloud deployments, an S3-compatible store is available in the commercial edition.

---

## 7 — Add custom metrics

```python
from typing import Any
from ayn_ml.metrics import register_metric
from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import DataSchema
from ayn_ml.core.spec import MetricSpec, MetricType

@register_metric("business_accuracy")
class BusinessAccuracyMetric:
    """Accuracy weighted by business cost per error type."""

    name = "business_accuracy"
    metric_type = MetricType.custom
    requires_reference = False

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        ...
```

---

## Next steps

- [Schemas reference](schemas.md) — `TabularSchema`, `TextSchema`, `AgentSchema`
- [Metrics reference](metrics.md) — all 71 built-in metrics
- [Metric Advisor](advisor.md) — routing rules, imbalance handling, API reference
- [Data Layer](data-layer.md) — `DataSource`, `SamplingStrategy`, `DataPartitioner`
- [Stores](stores.md) — `InMemoryStore`, `SqliteStore`
- [Architecture](architecture.md) — full design overview
