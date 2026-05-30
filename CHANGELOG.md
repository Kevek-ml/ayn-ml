# Changelog

## Unreleased

### New features

- **`ExcelSource(path, backend, sheet_name, read_kwargs)` added to `ayn_ml.data`.**
  Reads a worksheet from an Excel file and projects it to the columns required by the
  monitoring plan, matching the `DataFrameSource` contract.  Requires the opt-in extra:
  `pip install ayn-ml[excel]`.  `backend` accepts `"polars"` (uses `fastexcel`),
  `"pandas"` (uses `openpyxl`), or `"auto"` (tries Polars first, falls back to pandas).
  `sheet_name` is a first-class field — pass a string to select by name or a 0-based
  integer to select by position (pandas convention; translated automatically for the
  Polars backend).  `read_kwargs` are forwarded verbatim to the native reader — use
  kwargs that match your chosen backend's API.  When `backend="auto"`, a warning is
  logged naming the selected backend if `read_kwargs` is non-empty, so you can pin it
  explicitly.

- **`CsvSource(path, backend, separator, read_kwargs)` added to `ayn_ml.data`.**
  Reads a CSV file from disk and projects it to the columns required by the
  monitoring plan, matching the `DataFrameSource` contract.  `backend` accepts
  any narwhals-supported eager backend (`"polars"`, `"pandas"`, `"modin"`,
  `"cudf"`, `"pyarrow"`); the default `"auto"` tries Polars first and falls back
  to pandas.  `separator` is a first-class field normalised by narwhals across
  all backends.  `read_kwargs` are forwarded verbatim to the native reader —
  use kwargs that match your chosen backend's API.

### Bug fixes

- **`ttest` default changed: Welch's is now the default** (`equal_var=False`).
  Previously the default was Student's t-test (`equal_var=True`).  Welch's is
  safer in production because it does not assume equal variances.  If you rely
  on the Student's variant, pass `params={"equal_var": True}` explicitly.

- **`_MIN_ROWS` for statistical tests raised from 5 to 10.**  Pipelines
  computing `ks_2samp`, `ttest`, `mannwhitney`, `levene`, or `cramervonmises`
  on windows with fewer than 10 rows will now raise `InsufficientDataError`.
  This threshold is consistent with `drift.py` and avoids unreliable p-values
  on near-empty windows.

- **`auc` / `aucpr` now raise `MetricComputeError` instead of propagating a
  raw `ValueError`** when the current window contains only one class.

- **`r2` now raises `MetricComputeError` instead of returning `NaN`** when all
  `y_true` values are identical (zero variance).

- **`mape` now excludes near-zero true values** (`|y_true| <= eps`, default
  `eps=1e-8`) and emits a `logging.warning` listing the count of excluded rows.
  Previously only exact-zero values were excluded.  The `eps` threshold is
  configurable via `spec.params["eps"]`.

- **`mmd` is now deterministic by default** (`random_state=42`).  Previously
  the subsampling RNG was seeded with `None`, producing different results
  across runs.  Pass `params={"random_state": None}` to restore the old
  non-deterministic behaviour.

- **`psi` now uses union bins** (edges derived from both reference and current
  values) so out-of-range current values are included rather than silently
  dropped.  When current values fall outside the reference range a warning is
  emitted with the out-of-range count, the eps sensitivity range, and a
  suggestion to cross-check with `wasserstein` or `ks_2samp`.

- **`count` returns `0` instead of raising `InsufficientDataError`** on an
  empty window.  This makes it safe to use as a data-volume guard.

### Breaking changes

- **Schema column defaults changed to `None`.**  `BaseSchema.timestamp_col`,
  `model_id_col`, and `model_version_col` previously defaulted to
  `"timestamp"`, `"model_id"`, and `"model_version"`.  They now default to
  `None`.  Declare only the columns that exist in your DataFrame; the Runner
  validates their presence at runtime.  Migration: if you relied on the old
  defaults, set them explicitly — e.g.
  `TabularSchema(timestamp_col="timestamp", model_id_col="model_id")`.

- **`MonitoringPlan.partitioning` removed.**  The `partitioning` field (and
  `PartitioningConfig`) has been removed from `MonitoringPlan`.  Pass
  `reference` directly to `Runner.run()`.  Use `plan.window` to select the
  current window from a larger DataFrame.

- **`Runner(strict=True)` by default.**  The Runner now validates that every
  column declared in `data_schema` and every `MetricSpec.feature_name` is
  present in the DataFrame before executing any metric.  Raises `SchemaError`
  on failure.  Use `Runner(strict=False)` to restore lenient behaviour (warn
  and degrade gracefully).

### New fields

- **`ExecutionContext.run_id: str`** — hex UUID auto-generated per run.
  Stable within a run; use it to group all metric rows from the same
  `MonitoringReport` when reading from a store.  Also broadcast in
  `MonitoringReport.to_dataframe()`.

- **`ExecutionContext.n_current: int | None`** — row count of the current
  window after all filtering (windowing, sampling, model filtering).

- **`ExecutionContext.n_reference: int | None`** — row count of the reference
  window, or `None` when no reference was provided.

- **`Runner(n_jobs=..., strict=...)`** — `strict` parameter controls upfront
  column validation.  `n_jobs` enables parallel metric execution via
  `ThreadPoolExecutor`.

- **`MetricResult.effect_size: float | None`** — standardised effect size
  attached by statistical test metrics:
  - `ks_2samp` → KS D-statistic
  - `ttest` → Cohen's d (df-weighted pooled SD formula)
  - `mannwhitney` → Cliff's delta (in [-1, 1])
  - `levene` → variance ratio (cur / ref); `None` when reference variance is 0
  - `cramervonmises` → `None` (no standard effect size defined)
  The field is also exposed in `MonitoringReport.to_dataframe()`.
