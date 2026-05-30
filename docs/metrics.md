# Tabular Metrics Reference

All metrics are invoked via `get_metric(name).compute(cur, ref, schema, spec)` and
return a `MetricResult` with fields `val`, `status`, `effect_size`, and `effect_size_label`.

## Quick ref

### Performance

| Name | What it measures | Requires ref | sklearn required |
|---|---|---|---|
| `accuracy` | Fraction of correct predictions | No | Yes |
| `precision` | TP / (TP + FP) | No | Yes |
| `recall` | TP / (TP + FN) | No | Yes |
| `f1` | Harmonic mean of precision and recall | No | Yes |
| `log_loss` | Cross-entropy loss | No | Yes |
| `auc` | Area under the ROC curve | No | Yes |
| `aucpr` | Area under the precision-recall curve | No | Yes |
| `brier` | Mean squared err of calibrated probabilities | No | Yes |
| `mse` | Mean squared err | No | No |
| `mae` | Mean absolute err | No | No |
| `r2` | Coefficient of determination | No | No |
| `mape` | Mean absolute percentage err | No | No |

### Drift — distribution distance

| Name | What it measures | Numeric | Categorical | Requires `feature_name` |
|---|---|---|---|---|
| `psi` | Population Stability Index on a feature | Yes | Yes | Yes |
| `wasserstein` | Earth Mover's Distance on a feature | Yes | No | Yes |
| `mmd` | Maximum Mean Discrepancy (RBF kernel) on a feature | Yes | No | Yes |
| `target_drift` | PSI on `schema.label_col` — detects shifts in P(Y) | Yes | Yes | No |
| `hellinger` | Hellinger distance [0, 1] — histogram/category-based | Yes | Yes | Yes |
| `jensenshannon` | Jensen-Shannon distance [0, 1] — symmetric KL divergence | Yes | Yes | Yes |
| `tvd` | Total Variation Distance [0, 1] — 0.5 × Σ\|p−q\| | Yes | Yes | Yes |
| `energy_distance` | Energy statistics distance [0, ∞) | Yes | No | Yes |

### Drift — statistical tests

| Name | What it tests | Effect size | `effect_size_label` |
|---|---|---|---|
| `ks_2samp` | CDF equality (KS statistic) | KS D-statistic | `ks_statistic` |
| `ttest` | Mean equality (Welch's by default) | Cohen's d | `cohen_d` |
| `mannwhitney` | Rank distribution equality | Cliff's delta | `cliff_delta` |
| `levene` | Variance equality | Variance ratio | `variance_ratio` |
| `cramervonmises` | CDF equality (CvM statistic) | — | — |
| `anderson_darling` | k-sample Anderson-Darling — tail-sensitive | AD statistic | `ad_statistic` |
| `epps_singleton` | Frequency-domain (characteristic function) test | ES statistic | `es_statistic` |
| `fisher_exact` | Exact 2×2 test — binary only | Odds ratio | `odds_ratio` |
| `gtest` | G-test (log-likelihood ratio) — categorical/binary | Cramér's V | `cramer_v` |
| `ztest_proportions` | Two-proportion z-test — binary only | z-score | `z_score` |

### Statistics — descriptive

| Name | What it computes | Requires `feature_name` |
|---|---|---|
| `mean` | Arithmetic mean | Yes |
| `median` | 50th percentile | Yes |
| `std` | Sample standard deviation (ddof=1) | Yes |
| `skewness` | Fisher's skewness | Yes |
| `kurtosis` | Excess kurtosis (Fisher's, normal = 0) | Yes |
| `quantile` | Arbitrary quantile | Yes |
| `count` | Row count (including nulls) | Yes |
| `top_category` | Most frequent category (frequency as val) | Yes |
| `sum` | Sum of all values — numeric/binary | Yes |
| `unique_count` | Count of distinct values | Yes |
| `in_range_count` | Count of values in \[low, high\] — params: `low`, `high` | Yes |
| `out_range_count` | Count of values outside \[low, high\] — params: `low`, `high` | Yes |
| `in_list_count` | Count of values matching a list — param: `values` | Yes |
| `row_count` | Total row count of the DataFrame | No (`feature_name=None`) |
| `column_count` | Total column count of the DataFrame | No (`feature_name=None`) |
| `almost_constant_columns` | Count of columns with ≤ n_unique distinct values — param: `n_unique` (default 1) | No (`feature_name=None`) |
| `duplicate_rows` | Count of duplicate rows (total − unique rows) | No (`feature_name=None`) |
| `empty_columns` | Count of columns that are entirely null | No (`feature_name=None`) |

### Performance estimation — without ground truth (CBPE)

| Name | What it estimates | Requires ref | sklearn required |
|---|---|---|---|
| `cbpe_accuracy` | Estimated accuracy | Yes | Yes |
| `cbpe_auc` | Estimated ROC AUC | Yes | Yes |
| `cbpe_f1` | Estimated F1 score | Yes | Yes |
| `cbpe_precision` | Estimated precision | Yes | Yes |
| `cbpe_recall` | Estimated recall | Yes | Yes |

### Fairness

| Name | What it measures | Requires `feature_name` | Requires ref |
|---|---|---|---|
| `demographic_parity` | Max difference in positive prediction rates across groups | Yes | No |
| `equalized_odds` | Max gap in TPR and FPR across groups | Yes | No |
| `disparate_impact` | Ratio of lowest to highest positive prediction rate | Yes | No |
## Performance metrics

All performance metrics read column names from `TabularSchema`
(`label_col`, `prediction_col`, `proba_col`). They do not require a ref
window — they evaluate only the cur batch.

### `accuracy`

```
accuracy = correct predictions / total predictions
```

Reads `schema.label_col` and `schema.prediction_col`. Delegates to
`sklearn.metrics.accuracy_score`.

### `precision`, `recall`, `f1`

```
precision = TP / (TP + FP)
f1        = 2 × precision × recall / (precision + recall)
```

- `average` (default `"weighted"`) — averaging strategy for multi-class:
  `"binary"`, `"micro"`, `"macro"`, `"weighted"`, `"samples"`.

Delegates to `sklearn.metrics.precision_score`, `recall_score`, `f1_score`.

### `log_loss`

```
log_loss = -1/n × Σ [y log(p) + (1-y) log(1-p)]
```

Reads `schema.proba_col`. Lower is better. Sensitive to overconfident predictions.
Delegates to `sklearn.metrics.log_loss`.

### `auc`

Area under the ROC curve. Reads `schema.proba_col`.

- `multi_class` (default `"raise"`) — pass `"ovr"` or `"ovo"` for multi-class.

Raises `MetricComputeError` if the cur batch contains only one class
(ROC curve undefined).

### `aucpr`

Area under the precision-recall curve. More informative than AUC on
imbalanced datasets. Reads `schema.proba_col`.

Raises `MetricComputeError` if the cur batch contains only one class.

### `brier`

```
brier = 1/n × Σ (p_i - y_i)²
```

Proper scoring rule for calibration. Range [0, 1]; lower is better.
Reads `schema.proba_col`. Delegates to `sklearn.metrics.brier_score_loss`.

### `mse`, `mae`

Standard regression errors. Read `schema.label_col` and
`schema.prediction_col`.

```
mse = mean((y_true - y_pred)²)
mae = mean(|y_true - y_pred|)
```

### `r2`

```
r2 = 1 - SS_res / SS_tot
```

Range (-∞, 1]; 1 = perfect fit, 0 = mean-only baseline.
Raises `MetricComputeError` when all `y_true` values are identical
(`SS_tot = 0`; R² is undefined in that case — not 1.0).

### `mape`

```
mape = mean(|y_true - y_pred| / |y_true|) × 100
```

**Note:** MAPE is undefined when `y_true` is zero or near-zero. Rows where
`|y_true| ≤ eps` are excluded from the mean and a warning is logged.

- `eps` (default `1e-8`) — exclusion threshold for near-zero `y_true` values.
## Drift — distribution distance

All distance metrics require a ref window and operate on a single
feature column (`spec.feature_name`).

### `psi` — Population Stability Index

```
PSI = Σ (p_cur - p_ref) × ln(p_cur / p_ref)
```

Measures how much a feature's distribution has shifted. Supports both
numeric and categorical features.

**Numeric features:** bin edges are derived from the **union** of ref
and cur values so no data points are silently dropped. When cur
values fall outside the ref range, those bins will have
`p_ref ≈ 0` after clipping — their contribution is valid but sensitive to
the `eps` param. A warning is emitted with the out-of-range count, the
sensitivity range, and a suggestion to cross-check with `wasserstein` or
`ks_2samp`.

**Categorical features:** bins are the union of all categories seen in
either window; new categories in cur receive the full treatment without
out-of-range warnings.

- `n_bins` (default `10`) — number of bins for numeric features.
- `eps` (default `1e-4`) — zero-clipping floor applied before the log ratio.
  Controls the PSI contribution of empty ref bins. Changing this by
  one order of magnitude shifts the out-of-range bin contribution by
  ~0.7 units — see the warning message for the exact sensitivity range.

**Interpretation (rule of thumb):**

- PSI < 0.1 — stable
- 0.1 ≤ PSI < 0.25 — moderate drift
- PSI ≥ 0.25 — significant drift

**Limitation:** PSI on out-of-range bins is sensitive to `eps`. When the
warning fires, treat the absolute PSI value with caution and validate with
`wasserstein` or `ks_2samp`.

### `target_drift`

PSI on `schema.label_col` — measures how much P(Y) has shifted between
ref and cur windows. Does **not** require `spec.feature_name`.

**Target type inference** (override with `params["treat_as"]`):

- Integer or str labels → categorical PSI (class frequencies)
- Float labels → numeric PSI (histogram binning)

- `n_bins` (default `10`) — bins for numeric (regression) targets.
- `eps` (default `1e-4` numeric, `1e-8` categorical) — zero-clipping floor.
- `treat_as` (default `"auto"`) — `"numeric"` | `"categorical"` to override dtype inference.

Same PSI interpretation thresholds as `psi`:

- PSI < 0.1 — stable
- 0.1 ≤ PSI < 0.25 — moderate drift
- PSI ≥ 0.25 — significant drift

**Known limitation:** insensitive to symmetric concept drifts (e.g. boundary
inversion on a balanced dataset leaves P(Y=1) ≈ 0.5, so PSI ≈ 0).
See the [CBPE notebook](https://github.com/Kevek-ml/ayn-ml/blob/main/examples/03_cbpe.ipynb) for scenarios B and D.

### `wasserstein` — Earth Mover's Distance

Minimum "work" needed to transform the ref distribution into the
cur distribution. Units are the same as the feature (e.g., "the age
distribution shifted by 3.4 years on average"). Numeric only.

No natural threshold — set one based on the feature's domain (e.g.,
`threshold=0.05` for a probability score, `threshold=5.0` for age in years).
Delegates to `scipy.stats.wasserstein_distance`.

### `mmd` — Maximum Mean Discrepancy

Kernel two-sample statistic. Detects subtle shape differences that
histogram-based measures can miss (e.g., variance shifts without mean
shift). Numeric only.

- `max_samples` (default `500`) — caps both arrays before kernel computation
  to bound O(n²) cost.
- `random_state` (default `42`) — seed for reproducible subsampling.

Kernel bandwidth is estimated via the median heuristic on the ref data.
All tests return the two-tailed **p-value** as `val`. For threshold-based
alerting, set `spec.threshold` and `spec.upper_bound=True` to fire when
`p-value < threshold` (e.g., `threshold=0.05`).
All tests require a ref window and `spec.feature_name`.

Effect sizes are attached to `MetricResult.effect_size` alongside a label in
`MetricResult.effect_size_label` that identifies the scale. This allows
downstream code to interpret the number correctly without knowing which
metric produced it.

| Metric | `effect_size_label` | Scale | Rule of thumb |
|---|---|---|---|
| `ks_2samp` | `ks_statistic` | [0, 1] | No universal rule; > 0.1 often notable |
| `ttest` | `cohen_d` | (−∞, +∞), sign = direction | < 0.2 small · < 0.5 medium · ≥ 0.8 large |
| `mannwhitney` | `cliff_delta` | [−1, 1] | < 0.11 small · < 0.28 medium · ≥ 0.43 large |
| `levene` | `variance_ratio` | (0, +∞) | 1.0 = no change · > 1.5 or < 0.67 notable |
| `cramervonmises` | — | — | No effect size |
Effect size and p-value complement each other:
- **Low p-value + small effect** = statistically significant but operationally
  negligible (common with large n). Do not alert.
- **Large effect + high p-value** = practically meaningful shift but too few
  observations to confirm statistically. Investigate the sample.
- **Both low p-value and large effect** = genuine drift.
### `ks_2samp` — Kolmogorov-Smirnov two-sample test

Tests whether two samples come from the same continuous distribution.
Sensitive to location, scale, and shape differences.

`effect_size` = KS D-statistic = max absolute difference between the
two empirical CDFs. Range [0, 1].

Delegates to `scipy.stats.ks_2samp`.

### `ttest` — Welch's t-test (default) / Student's t-test

Parametric test for equality of means. Assumes approximately normal data.

**Default is Welch's (`equal_var=False`)** — safe when the two groups have
different variances. Student's t-test (`equal_var=True`) is slightly more
powerful when variances are provably equal, but biased otherwise.

- `equal_var` (default `False`) — set to `True` for Student's t-test.

`effect_size` = Cohen's d using the df-weighted pooled standard deviation:

```
pooled_std = sqrt(((n1-1)×s1² + (n2-1)×s2²) / (n1+n2-2))
cohen_d    = (mean_ref - mean_cur) / pooled_std
```

Sign convention: positive d means the ref mean is higher than cur.
Delegates to `scipy.stats.ttest_ind`.

### `mannwhitney` — Mann-Whitney U test

Non-parametric alternative to the t-test. Compares rank distributions;
robust to non-normality and outliers. Prefer over `ttest` when
`|skewness| > 1.0` or for ordinal data.

- `alternative` (default `"two-sided"`) — `"less"` or `"greater"` for one-sided tests.

`effect_size` = Cliff's delta:

```
cliff_delta = 2U / (n_ref × n_cur) − 1
```

Range [−1, 1]. Positive = ref tends to be larger than cur.
Delegates to `scipy.stats.mannwhitneyu`.

### `levene` — Levene's test for variance equality

Tests whether variances (scale) have changed between ref and cur.
Less sensitive to departures from normality than Bartlett's test.

`effect_size` = variance ratio:

```
variance_ratio = var_cur / var_ref   (ddof=1)
```

Values above 1 mean cur has more spread; below 1 means less.
`None` when `var_ref = 0` (constant ref).
Delegates to `scipy.stats.levene`.
### `cramervonmises` — Cramér-von Mises two-sample test

Tests CDF equality by integrating squared CDF differences — more powerful
than `ks_2samp` for sustained shifts across the full distribution (e.g.,
a location shift without a spike). No effect size.

Delegates to `scipy.stats.cramervonmises_2samp`.

### `anderson_darling` — Anderson-Darling k-sample test

More sensitive to tail differences than the KS test. Uses `scipy.stats.anderson_ksamp`.
Returns a p-value (scipy approximation clamps to [0.001, 0.25]).

- Accepts: numeric, binary.
- `effect_size` = AD test statistic.

```python
MetricSpec(name="anderson_darling", feature_name="age", threshold=0.05)
```

### `epps_singleton` — Epps-Singleton two-sample test

Frequency-domain test based on the empirical characteristic function. Generally more powerful
than KS for alternatives where the CDFs are close but the characteristic functions differ
(e.g. different shapes). Delegates to `scipy.stats.epps_singleton_2samp`.

- Accepts: numeric, binary.
- `effect_size` = test statistic (chi-square distributed under H₀ with 25 degrees of freedom).

### `hellinger` — Hellinger distance

Symmetric, bounded [0, 1]: 0 = identical distributions, 1 = fully disjoint.
Builds a shared histogram for numeric features; uses category counts for categorical features.
Higher values indicate more drift — use an upper-bound threshold.

- Accepts: numeric, binary, categorical.
- `spec.params["bins"]` — overrides adaptive bin count (default: `min(max(sqrt(n), 10), 100)`).

```python
MetricSpec(name="hellinger", feature_name="age", threshold=0.1)
```

### `jensenshannon` — Jensen-Shannon distance

Symmetric square root of the JS divergence, bounded [0, 1]. Delegates to
`scipy.spatial.distance.jensenshannon`. Builds a shared histogram for numeric features;
uses category counts for categorical features. Higher values indicate more drift.

- Accepts: numeric, binary, categorical.
- `spec.params["bins"]` — overrides adaptive bin count.

### `tvd` — Total Variation Distance

TVD = 0.5 × Σ|p_i − q_i|, bounded [0, 1]. Builds a shared histogram for numeric features;
uses category counts for categorical features. Higher values indicate more drift.

- Accepts: numeric, binary, categorical.
- `spec.params["bins"]` — overrides adaptive bin count.

### `energy_distance` — Energy distance

Based on `scipy.stats.energy_distance`. Returns a non-negative distance (no upper bound);
0 = identical distributions. More sensitive to tail differences than the KS test.

- Accepts: numeric, binary only.

### `fisher_exact` — Fisher's exact test

Exact test for association between two binary outcomes in a 2×2 contingency table.
Preferred over chi-square when any expected cell frequency is below 5.

- Accepts: binary only.
- `effect_size` = odds ratio.
- `spec.params["alternative"]` — `"two-sided"` (default), `"less"`, or `"greater"`.

```python
MetricSpec(name="fisher_exact", feature_name="is_fraud", threshold=0.05)
```

### `gtest` — G-test (log-likelihood ratio)

Goodness-of-fit test based on the log-likelihood ratio. Returns a p-value alongside
Cramér's V as effect size. Appropriate for categorical and binary features.

- Accepts: categorical, binary.
- `effect_size` = Cramér's V.
- `effect_size_label` = `"cramer_v"`.

### `ztest_proportions` — Two-proportion z-test

Two-proportion z-test comparing the proportion of 1s in the reference vs. current window.
Returns a p-value alongside a z-score as effect size.

- Accepts: binary only.
- `effect_size` = z-score.
- `effect_size_label` = `"z_score"`.

**Threshold table for new drift metrics:**

| Metric | Value range | Threshold convention | Alert when |
|---|---|---|---|
| `hellinger` | [0, 1] | upper bound | value > threshold |
| `jensenshannon` | [0, 1] | upper bound | value > threshold |
| `tvd` | [0, 1] | upper bound | value > threshold |
| `energy_distance` | [0, ∞) | upper bound | value > threshold (domain-dependent) |
| `anderson_darling` | p-value | lower bound | p < threshold (e.g. 0.05) |
| `epps_singleton` | p-value | lower bound | p < threshold |
| `fisher_exact` | p-value | lower bound | p < threshold |
| `gtest` | p-value | lower bound | p < threshold |
| `ztest_proportions` | p-value | lower bound | p < threshold |

## Statistics — descriptive

Descriptive metrics do not require a ref window. Column-level metrics
(`mean` through `in_list_count`) require `spec.feature_name`. Dataset-level
metrics (`row_count`, `column_count`, `almost_constant_columns`, `duplicate_rows`,
`empty_columns`) operate on the full DataFrame and expect `spec.feature_name = None`.

### `mean`, `median`, `std`

Standard summary stats. `std` uses `ddof=1` (sample standard deviation).

### `skewness`

Fisher's skewness (third standardised moment). 0 = symmetric; > 0 =
right-tailed; < 0 = left-tailed. Rule of thumb: `|skewness| > 1` → prefer
`mannwhitney` over `ttest` for drift detection on this feature.

### `kurtosis`

Excess kurtosis (Fisher's definition; normal distribution = 0). High positive
values indicate heavy tails. Rule of thumb: `excess_kurtosis > 7` → prefer
`cramervonmises` over `ks_2samp`.

### `quantile`

- `q` (default `0.5`) — quantile probability in (0, 1). `q=0.5` returns
  the median; `q=0.95` returns the 95th percentile.

### `count`

Total row count of the feature column, including null values. Returns `0`
on an empty batch (no error).

### `top_category`

Most frequent category. `val` is the frequency (proportion in [0, 1]) of
the top category, not the category label itself.

Tie-breaking: when two categories share the same count, the lexicographically
earliest one (alphabetically first) is returned. This is deterministic but
arbitrary — do not rely on it for business logic.

### `sum`

Sum of all values in a numeric or binary feature column.

- Accepts: numeric, binary.
- No reference required. Supports optional threshold.

### `unique_count`

Count of distinct values in a feature column.

- Accepts: numeric, categorical, binary.
- No reference required.

### `in_range_count`

Count of values falling within a closed interval [low, high].

- Accepts: numeric, binary.
- `spec.params["low"]` — lower bound inclusive (default `-inf`).
- `spec.params["high"]` — upper bound inclusive (default `+inf`).

```python
MetricSpec(name="in_range_count", feature_name="score", params={"low": 0.0, "high": 1.0})
```

### `out_range_count`

Count of values falling outside the closed interval [low, high].

- Accepts: numeric, binary.
- Same `low` / `high` params as `in_range_count`.

### `in_list_count`

Count of values matching any entry in a reference list.

- Accepts: numeric, categorical, binary.
- `spec.params["values"]` — list of allowed values (default empty — returns 0).

```python
MetricSpec(name="in_list_count", feature_name="region", params={"values": ["A", "B", "C"]})
```

### Dataset-level statistics

The following five metrics operate on the entire DataFrame rather than a single column.
Set `spec.feature_name = None` (the default).

#### `row_count`

Total row count of the current window DataFrame.

```python
MetricSpec(name="row_count", threshold=1000, upper_bound=False)  # alert if < 1000 rows
```

#### `column_count`

Total column count of the current window DataFrame.

#### `almost_constant_columns`

Count of columns whose distinct-value count is at or below a threshold.

- `spec.params["n_unique"]` — maximum number of distinct values allowed (default `1`).
  Set to `2` to also flag near-constant binary columns.

```python
MetricSpec(name="almost_constant_columns", params={"n_unique": 2}, threshold=0)
```

#### `duplicate_rows`

Count of duplicate rows (total rows − unique rows). A row appearing 3 times contributes 2.

```python
MetricSpec(name="duplicate_rows", threshold=0)  # alert when any duplicates present
```

#### `empty_columns`

Count of columns that are entirely null (all values are `None` / `NaN`).

```python
MetricSpec(name="empty_columns", threshold=0)  # alert when any column is all-null
```

## Performance estimation — without ground truth (CBPE)

Confidence-Based Performance Estimation estimates classification metrics on
unlabelled windows. No ground-truth labels are needed on the cur window —
the ref window (which has labels) is used only to fit a probability calibrator.

**Core idea:** if a model assigns p̂ = 0.9 to an observation, there is a 90%
chance that observation is correctly classified. Summing these fractional
certainties reconstructs an estimated confusion matrix and derived metrics.

**Reference:** Vandewiele et al., NannyML (2022).

### Schema requirements

- `TabularSchema.proba_col` — must be set and present on **both** windows.
- `TabularSchema.label_col` — must be present on the **ref** window only.
- `TabularSchema.prediction_col` — required for `cbpe_accuracy`, `cbpe_f1`,
  `cbpe_precision`, `cbpe_recall`; not used by `cbpe_auc`.

Both windows require at least 100 rows. The calibrator needs enough ref
data to fit reliably; estimates on small windows are noisy.

### `cbpe_accuracy`

```
estimated_accuracy = (Σ p̂ᵢ [y_pred=1]  +  Σ (1−p̂ᵢ) [y_pred=0]) / N
```

Element-wise weighted sum of fractional correct predictions divided by total rows.
Each row contributes `p̂ᵢ` if the model predicted positive (probability it is correct)
or `1−p̂ᵢ` if predicted negative.

### `cbpe_auc`

Estimated ROC AUC via cumulative sum over calibrated probabilities sorted
descending. O(n log n).

Raises `MetricComputeError` if estimated positive or negative mass is near
zero (single-class degenerate case).

### `cbpe_f1`, `cbpe_precision`, `cbpe_recall`

Estimated from a fractional confusion matrix:

```
estimated_TP = Σ p̂_i  for i where y_pred = 1
estimated_FP = Σ (1 - p̂_i)  for i where y_pred = 1
estimated_FN = Σ (1 - p̂_i)  for i where y_pred = 0
```

| param | default | effect |
|---|---|---|
| `calibrate` | `True` | Fit isotonic regression calibrator on ref before estimating. Set to `False` to use raw probabilities (faster but less accurate if the model is miscalibrated). |

**Known limitations:**

- **Concept drift blindness**: if the relationship P(Y\|X) changes but the
  model's probabilities do not reflect it, CBPE will report stable performance.
  Always pair with drift metrics (`psi`, `wasserstein`) to detect covariate drift.
- **Calibration assumption**: CBPE assumes the model is well-calibrated (after
  isotonic correction on the ref). A badly miscalibrated model will
  produce unreliable estimates even with `calibrate=True`.
- **Binary only**: all five metrics assume binary classification.
## Fairness metrics

Fairness metrics measure group-level disparity in model predictions. They
operate on the cur window only (no ref required) and identify the
protected attribute via `spec.feature_name`.

Declare sensitive attributes in `TabularSchema.protected_cols` to enable
early validation (the Runner rejects an unknown column before any metric runs):

```python
schema = TabularSchema(
    label_col="y_true",
    prediction_col="y_pred",
    protected_cols=["gender", "age_group"],
)
```

When `protected_cols` is `None`, the declaration check is skipped but the
column must still exist in the DataFrame.

### Using `spec.feature_name`

Each fairness `MetricSpec` names the protected column via `feature_name`,
exactly as drift and stats metrics do:

```python
MetricSpec(name="demographic_parity", feature_name="gender", threshold=0.1)
MetricSpec(name="equalized_odds",     feature_name="gender", threshold=0.1)
MetricSpec(name="disparate_impact",   feature_name="gender",
           threshold=0.8, upper_bound=False)
```

### `demographic_parity` — Demographic Parity Difference

```
DPD = max_g P(Ŷ=1 | A=g) − min_g P(Ŷ=1 | A=g)
```

Maximum difference in positive prediction rates between any two groups.
Range [0, 1]. Lower is better. A value of 0 means perfect parity.

**Threshold guidance:** values above 0.1 are worth investigating; above 0.2
indicate substantial disparity.

Uses `schema.prediction_col` only — no ground-truth labels required.

### `equalized_odds` — Equalized Odds Difference

```
EOD = max(max_g TPR_g − min_g TPR_g,  max_g FPR_g − min_g FPR_g)
```

Takes the larger of the TPR gap and FPR gap across groups. Range [0, 1].
Lower is better. A value of 0 means all groups have identical true positive
and false positive rates.

Groups with no positive (or no negative) labels in the cur window
contribute `nan` to their TPR (or FPR) and are excluded from the gap via
`nanmax` / `nanmin`.

Uses both `schema.label_col` and `schema.prediction_col`.

**Threshold guidance:** same as demographic parity — 0.1 as a soft limit.

### `disparate_impact` — Disparate Impact Ratio

```
DIR = min_g P(Ŷ=1 | A=g) / max_g P(Ŷ=1 | A=g)
```

Ratio of the lowest to the highest positive prediction rate. Range [0, 1].
Higher is better (closer to 1.0 = no disparity).

**80% rule (EEOC guideline):** a DIR below 0.8 is considered potentially
discriminatory. To apply it:

```python
MetricSpec(
    name="disparate_impact",
    feature_name="gender",
    threshold=0.8,
    upper_bound=False,   # passes when val >= threshold
)
```

Returns 1.0 (no disparity) when all groups have a near-zero positive
prediction rate to avoid division-by-zero.

## Recommender-system metrics

All recsys metrics require a `RecSysSchema` and an interactions DataFrame — one row per
`(user_id, item_id)` pair.

```python
from ayn_ml.core.schema import RecSysSchema
from ayn_ml.core.spec import MetricSpec, MonitoringPlan

schema = RecSysSchema(
    user_id_col="user_id",
    item_id_col="item_id",
    relevance_col="relevance",
    score_col="score",             # predicted score column
    recommendations_type="score",  # "score" | "rank"
)
```

If your data is in a **user×item matrix**, convert it first:

```python
from ayn_ml.metrics.recsys_utils import interactions_from_matrix

interactions = interactions_from_matrix(truth_matrix, pred_matrix)
```

All metrics accept `params["k"]` (cutoff, default 10) and `params["relevance_threshold"]`
(items with `relevance > threshold` count as relevant, default 0).

---

### Ranking accuracy

#### `precision_at_k`

Mean fraction of the top-K recommended items that are relevant.

`P@K = |relevant ∩ top_k| / K` — averaged across users.

#### `recall_at_k`

Mean fraction of relevant items that appear in the top-K list.

`R@K = |relevant ∩ top_k| / |relevant|` — averaged across users. Users with no relevant items contribute 0.

#### `fbeta_at_k`

Harmonic mean of `precision_at_k` and `recall_at_k` weighted by `beta` (default 1.0 → F1).

Extra param: `params["beta"]` (float, default 1.0).

#### `hit_rate`

Fraction of users for whom at least one relevant item appears in the top-K list.

#### `map_at_k`

Mean Average Precision@K. For each user the average precision is the mean of `P@i` at every
position `i` where a relevant item appears, normalized by the total number of relevant items.

!!! note
    Normalization uses `|relevant|`, not `min(|relevant|, k)`. When a user has more relevant
    items than `k`, a perfect top-K still yields `AP@K < 1.0`.

#### `ndcg_at_k`

Normalized Discounted Cumulative Gain@K. Discounts gains by `log₂(rank + 1)`, normalized against
the ideal ranking. Supports **graded relevance** — raw relevance column values are used as gains.

#### `mrr_at_k`

Mean Reciprocal Rank@K. Reciprocal of the rank of the first relevant item (`1 / rank`), averaged
across users. Returns 0 for users with no relevant item in the top-K.

---

### Beyond-accuracy metrics

#### `diversity`

Mean intra-list diversity: average pairwise cosine distance between item feature vectors within
each user's top-K list, averaged across users. Range `[0, 1]`.

Requires `params["item_features"]`: list of column names present in `current` (e.g.
`["genre_embedding_0", "genre_embedding_1"]`).

#### `novelty`

Mean `−log₂(pop(i))` where `pop(i)` is the fraction of **training interactions** involving
item `i`. Higher = less popular items recommended. Requires `reference` = training interactions DataFrame.

#### `popularity_bias`

Mean item popularity of recommended items. Opposite of novelty — higher = more popular items
recommended. Requires `reference` = training interactions DataFrame.

#### `personalization`

`1 − mean pairwise cosine similarity` between users' binary top-K indicator vectors. Range `[0, 1]`:
0 = all users receive identical lists; 1 = all lists are completely disjoint.

#### `item_bias`

Gini coefficient of item recommendation frequency across all users' top-K lists. Range `[0, 1)`:
0 = all items recommended equally; approaching 1 = a few items dominate all slots.

#### `user_bias`

Gini coefficient of per-user recommendation list length. Detects cold-start users who receive
shorter lists than `k` due to sparse item catalogues. Range `[0, 1)`.

#### `serendipity`

Mean serendipity@K across users. For each item in the top-K list:
`serendipity(i, u) = relevance(i, u) × unexpectedness(i, u)`, where unexpectedness is the
cosine distance between the item's feature vector and the centroid of items the user interacted
with in the training data. Range `[0, 1]`.

Requires `reference` (training interactions DataFrame) and `params["item_features"]`
(list of numeric feature column names present in `current`).

---

## NLP / LLM *(planned — pip install ayn-ml[nlp])*
`bleu` `rouge1` `rouge2` `rougeL` `bert_score` `exact_match` `embedding_drift`

---

## Premium Metrics — `ayn-ml-pro`

> **Status:** All premium metrics are registered automatically when `ayn_ml_pro.metrics` is imported. Individual metric implementations are not yet complete — registration placeholders are in place.

Premium metrics are registered into the `ayn-ml` metric registry on import. Once registered, use them via `MetricSpec` exactly like any built-in metric.

```python
import ayn_ml_pro.metrics  # registers all premium metrics
from ayn_ml.core.spec import MetricSpec, MonitoringPlan

plan = MonitoringPlan(
    name="my_monitor",
    metrics=[MetricSpec(name="toxicity"), MetricSpec(name="llm_faithfulness")],
)
```

### Safety metrics

**Module:** `ayn_ml_pro.metrics.safety`

**Extra required:** `pip install "ayn-ml-pro[llm-openai]"` or `pip install "ayn-ml-pro[llm-anthropic]"`

| Metric name | Description |
|---|---|
| `toxicity` | Fraction of model outputs classified as toxic by a safety model |
| `hallucination_rate` | Fraction of outputs flagged as hallucinations against a grounding corpus |

### LLM-judge metrics

**Module:** `ayn_ml_pro.metrics.llm_judge`

**Extra required:** `pip install "ayn-ml-pro[llm-openai]"` or `pip install "ayn-ml-pro[llm-anthropic]"`

Scores model outputs using a language model judge. Suitable for RAG pipelines, chatbots, and generative models where traditional metrics are not applicable.

| Metric name | Description |
|---|---|
| `llm_relevance` | Relevance of the model output to the input prompt, scored by an LLM judge |
| `llm_faithfulness` | Factual faithfulness of the output to a grounding context, scored by an LLM judge |
| `llm_coherence` | Linguistic coherence and fluency score |

### Agent evaluation metrics

**Module:** `ayn_ml_pro.metrics.agent`

Evaluates agentic systems where the model takes sequences of actions (tool calls, reasoning steps) to complete a task.

> `task_completion_rate`, `tool_success_rate`, `step_count`, `avg_tokens`, `avg_latency_ms`, and `avg_cost_usd` are registered by the `ayn-ml` package — use them directly without importing `ayn_ml_pro.metrics`.

| Metric name | Description | Ground truth required |
|---|---|---|
| `step_efficiency` | Ratio of actual steps to optimal step count | Yes (ground-truth traces) |
| `goal_adherence` | LLM-judged score for how closely the agent followed the stated goal | No |

### LLM metric timeouts

LLM-backed metrics (safety, judge, agent `goal_adherence`) involve external API calls and may be slow. Set a timeout via `MetricSpec.params`:

```python
MetricSpec(name="llm_faithfulness", params={"timeout": 30})  # 30 seconds
```

If the call exceeds the timeout, the metric result is recorded as an error in `MonitoringReport.errors` — the run continues for all other metrics.
