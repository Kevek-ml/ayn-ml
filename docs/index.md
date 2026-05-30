---
hide:
  - navigation
  - toc
---

<div class="ayn-hero">
  <span class="ayn-hero__eyebrow">Open Source · Apache 2.0 License</span>
  <h1 class="ayn-hero__title">ML monitoring for <em>all three</em> modalities.</h1>
  <p class="ayn-hero__sub">
    Tabular classifiers, LLM pipelines, and AI agents — one library, one API,
    narwhals backend so it works with pandas and Polars out of the box.
  </p>
  <div class="ayn-install">
    <span class="ayn-install__label">$</span>
    pip install ayn-ml
  </div>
</div>

---

## Why ayn-ml?

Most monitoring libraries specialise in a single modality. ayn-ml covers tabular ML, NLP/LLM pipelines, and AI agents with a single, consistent API — declarative configuration, a protocol-extensible metric registry, and automatic metric selection via the built-in Advisor.

---

## Key features

<div class="ayn-grid">

<div class="ayn-card">
  <div class="ayn-card__icon">🧠</div>
  <p class="ayn-card__title">Auto metric selection</p>
  <p class="ayn-card__desc">
    <code>MetricAdvisor</code> inspects your data and builds the right monitoring plan — 
    normality tests, variance routing, imbalance detection — automatically.
  </p>
</div>

<div class="ayn-card">
  <div class="ayn-card__icon">⚡</div>
  <p class="ayn-card__title">narwhals backend</p>
  <p class="ayn-card__desc">
    Write once, run on pandas or Polars. No conversion overhead, no backend lock-in.
  </p>
</div>

<div class="ayn-card">
  <div class="ayn-card__icon">📐</div>
  <p class="ayn-card__title">Declarative config</p>
  <p class="ayn-card__desc">
    Define monitoring plans in Python or YAML. Plans are immutable Pydantic models 
    that round-trip perfectly.
  </p>
</div>

<div class="ayn-card">
  <div class="ayn-card__icon">🔌</div>
  <p class="ayn-card__title">Extensible registry</p>
  <p class="ayn-card__desc">
    <code>@register_metric</code> to add custom metrics. They integrate seamlessly with
    the Runner, stores, and renderers.
  </p>
</div>

<div class="ayn-card">
  <div class="ayn-card__icon">📊</div>
  <p class="ayn-card__title">71 built-in metrics</p>
  <p class="ayn-card__desc">
    Performance, drift, statistics, CBPE estimation, fairness, and recsys — all in one registry,
    all with consistent result types.
  </p>
</div>

<div class="ayn-card">
  <div class="ayn-card__icon">🚨</div>
  <p class="ayn-card__title">Alerts & renderers</p>
  <p class="ayn-card__desc">
    Threshold policies, email/webhook channels, and an HTML renderer with 
    snapshot and history dashboards.
  </p>
</div>

</div>

---

## Get started in 60 seconds

=== "Python"

    ```python
    from ayn_ml import MonitoringPlan, TabularSchema, MetricSpec
    from ayn_ml.runner import Runner

    schema = TabularSchema(label_col="y_true", prediction_col="y_pred")

    plan = MonitoringPlan(
        name="churn_monitor",
        model_id="churn_v2",
        model_version="1.0",
        data_schema=schema,
        metrics=[
            MetricSpec(name="accuracy", threshold=0.85, upper_bound=False),
            MetricSpec(name="psi", feature_name="age", threshold=0.2),
        ],
    )

    report = Runner().run(plan, current=df_current, reference=df_reference)
    print(report.to_dataframe())
    ```

=== "YAML (roadmap)"

    Plans are declarative Pydantic models — YAML round-trip (`from_yaml` / `to_yaml`)
    is on the roadmap. The schema below reflects the planned config format:

    ```yaml
    # churn_plan.yaml
    name: churn_monitor
    model_id: churn_v2
    model_version: "1.0"
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

=== "Auto (MetricAdvisor)"

    ```python
    from ayn_ml.advisor import MetricAdvisor
    from ayn_ml.core.schema import TabularSchema

    schema   = TabularSchema(label_col="y_true", prediction_col="y_pred", proba_col="y_prob")
    designer = MetricAdvisor(schema)

    result = designer.suggest(
        df_current,
        reference=df_reference,   # required — stable baseline
        task_type="classification",
    )

    plan  = result.plan      # MonitoringPlan — pass directly to Runner
    for w in result.warnings:
        print("Advisory:", w)
    ```

[Quick Start →](quickstart.md){ .md-button .md-button--primary }
[API Reference →](api/index.md){ .md-button }

---

## Installation

```bash
pip install ayn-ml                   # core (narwhals, pandas, Pydantic, scipy, sklearn)
pip install ayn-ml[polars]           # add Polars backend (recommended for production)
pip install ayn-ml[mlflow]           # MLflow store
pip install ayn-ml[nlp]              # BLEU, ROUGE, BERTScore
pip install ayn-ml[all]              # everything
```

> **Python ≥ 3.10 required.**

---

## Need more?

[ayn-ml Pro](pro.md) adds advanced advisor profiles, a cloud-scale runner, Slack notifications,
S3 storage, LLM safety and quality metrics, agent evaluation metrics, advanced explainability,
and an analytics dashboard with RBAC and SSO.

ayn-ml Pro is a commercial product by Kevek ML.
