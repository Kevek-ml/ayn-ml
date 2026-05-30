# Metrics

ayn-ml ships with 71 built-in metrics across five categories: tabular (performance, drift, statistics, CBPE estimation, fairness) and recsys.
All metrics are resolved from the registry by name at runtime.

---

## Registry

::: ayn_ml.metrics.registry.register_metric

::: ayn_ml.metrics.registry.list_metrics

::: ayn_ml.metrics.registry.get_metric

---

## Base protocol

::: ayn_ml.metrics.base.Metric

---

## Built-in metrics

### Performance

| Name | Description |
|---|---|
| `accuracy` | Classification accuracy |
| `f1` | F1 score (macro) |
| `precision` | Precision (macro) |
| `recall` | Recall (macro) |
| `auc` | ROC-AUC |
| `aucpr` | Area under precision-recall curve |
| `log_loss` | Log loss |
| `brier` | Brier score |
| `mae` | Mean absolute error (regression) |
| `mse` | Mean squared error (regression) |
| `r2` | R² score (regression) |
| `mape` | Mean absolute percentage error (regression) |

### Drift

| Name | Description |
|---|---|
| `psi` | Population Stability Index |
| `wasserstein` | Wasserstein distance (Earth Mover's Distance) |
| `cramervonmises` | Cramér-von Mises test |
| `ks` | Kolmogorov-Smirnov test |
| `ttest` | Welch's t-test |
| `mannwhitney` | Mann-Whitney U test |
| `levene` | Levene's test (variance shift) |
| `chisquare` | Chi-square test (categorical) |
| `mmd` | Maximum Mean Discrepancy |
| `target_drift` | Label distribution drift |

### Statistics

| Name | Description |
|---|---|
| `mean` | Column mean |
| `std` | Column standard deviation |
| `skewness` | Sample skewness |
| `kurtosis` | Excess kurtosis |
| `top_category` | Most frequent category and its frequency |
| `missing_rate` | Fraction of null values |

### CBPE Estimation

| Name | Description |
|---|---|
| `cbpe_accuracy` | CBPE-estimated accuracy (no ground truth required) |
| `cbpe_f1` | CBPE-estimated F1 |
| `cbpe_auc` | CBPE-estimated ROC-AUC |
| `cbpe_precision` | CBPE-estimated precision |
| `cbpe_recall` | CBPE-estimated recall |

### Fairness

| Name | Description |
|---|---|
| `demographic_parity` | Demographic parity difference |
| `equalized_odds` | Equalized odds difference |
| `disparate_impact` | Disparate impact ratio |

### Recsys

Operate on an interactions DataFrame (one row per user-item pair) described by a `RecSysSchema`.
All accept `params["k"]` (ranking cutoff, default 10) and `params["relevance_threshold"]` (default 0).

| Name | Description |
|---|---|
| `precision_at_k` | Mean Precision@K — fraction of top-K items that are relevant, averaged over users |
| `recall_at_k` | Mean Recall@K — fraction of relevant items found in top K, averaged over users |
| `fbeta_at_k` | Mean F-beta@K — harmonic mean of Precision@K and Recall@K; `params["beta"]` controls weight (default 1.0) |
| `hit_rate` | Mean Hit Rate@K — fraction of users with at least one relevant item in top K |
| `map_at_k` | Mean Average Precision@K — average of per-user AP@K scores |
| `ndcg_at_k` | Normalized Discounted Cumulative Gain@K — ranking quality with position discount; supports binary and graded relevance |
| `mrr_at_k` | Mean Reciprocal Rank@K — mean of the reciprocal rank of the first relevant item within top K |
| `diversity` | Mean intra-list cosine distance across users — requires `params["item_features"]` |
| `novelty` | Mean -log2(popularity) of recommended items — requires reference interaction log |
| `popularity_bias` | Mean item popularity of recommended items — requires reference interaction log |
| `personalization` | 1 - mean inter-user cosine similarity of recommendation lists |
| `item_bias` | Gini coefficient of item recommendation frequency |
| `user_bias` | Gini coefficient of per-user list length |
| `serendipity` | Mean serendipity@K — relevance × unexpectedness (cosine distance to user training profile) — requires `reference` and `params["item_features"]` |

::: ayn_ml.metrics.recsys.PrecisionAtKMetric

::: ayn_ml.metrics.recsys.RecallAtKMetric

::: ayn_ml.metrics.recsys.FBetaAtKMetric

::: ayn_ml.metrics.recsys.HitRateMetric

::: ayn_ml.metrics.recsys.MAPAtKMetric

::: ayn_ml.metrics.recsys.NDCGAtKMetric

::: ayn_ml.metrics.recsys.MRRAtKMetric

::: ayn_ml.metrics.recsys.DiversityMetric

::: ayn_ml.metrics.recsys.NoveltyMetric

::: ayn_ml.metrics.recsys.PopularityBiasMetric

::: ayn_ml.metrics.recsys.PersonalizationMetric

::: ayn_ml.metrics.recsys.ItemBiasMetric

::: ayn_ml.metrics.recsys.UserBiasMetric

::: ayn_ml.metrics.recsys.SerendipityMetric

---

## Recsys utilities

Helper to convert a wide user×item matrix into the long interactions table expected by all recsys metrics.

::: ayn_ml.metrics.recsys_utils.interactions_from_matrix

LLM safety metrics (toxicity, hallucination detection, LLM-as-a-judge scoring) are available in the commercial edition.
