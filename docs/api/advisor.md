# Advisor

`MetricAdvisor` analyses your data and generates a ready-to-use `MonitoringPlan`.
It routes each feature to statistically appropriate drift metrics based on column type,
sample size, normality, skewness, and variance relative to a reference window.

→ [Full advisor guide with decision trees](../advisor.md)

---

## MetricAdvisor

::: ayn_ml.advisor.MetricAdvisor

---

## SuggestedPlan

::: ayn_ml.advisor.SuggestedPlan

---

## Column analysis internals

!!! note "Internal API"
    `ColumnAnalysis` is an internal data class exposed here for transparency.
    Its fields and behaviour are not covered by semantic versioning guarantees.

::: ayn_ml.advisor._analysis.ColumnAnalysis
