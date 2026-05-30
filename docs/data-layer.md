# Data layer

The data layer sits between your raw data and the runner. It has three independent concerns — compose them as needed:

```
DataSource  →  WindowStrategy  →  RandomSampling  →  current
   load         select window      reduce size

DataSource  ─────────────────────────────────────→  reference
   load (separate — training baseline or frozen snapshot)
```

## Data sources

A `DataSource` loads a DataFrame and projects it to the minimal column set required by the plan. Column selection is automatic — driven by the schema's column declarations, any `feature_name` values in metric specs, and any `profile_cols` declared on the plan.

### `DataFrameSource` — in-memory DataFrame

```python
from ayn_ml.data.source import DataFrameSource, required_columns

source = DataFrameSource(df)
df_loaded = source.load(plan)   # projects to exactly what the plan needs
```

### `CsvSource` — CSV file on disk

`CsvSource` reads a CSV file from disk using any narwhals-supported eager backend
(`"polars"`, `"pandas"`, `"modin"`, `"cudf"`, `"pyarrow"`).
The default `backend="auto"` tries Polars first and falls back to pandas.

```python
from ayn_ml.data import CsvSource

# auto backend — Polars if installed, else pandas
source = CsvSource(path="data/production.csv")
df_loaded = source.load(plan)   # projects to exactly what the plan needs
```

**Separator** is a first-class parameter handled uniformly by narwhals — it works
correctly with every backend without any manual translation:

```python
source = CsvSource(path="data/production.csv", separator="|")
```

**Explicit backend selection** lets you pick the engine and use its native kwargs directly:

```python
# Polars backend — use Polars kwargs
source = CsvSource(path="data/production.csv", backend="polars", read_kwargs={"n_rows": 10_000})

# pandas backend — use pandas kwargs
source = CsvSource(path="data/production.csv", backend="pandas", read_kwargs={"nrows": 10_000})

# modin, cudf, pyarrow — same pattern
source = CsvSource(path="data/production.csv", backend="modin")
```

`read_kwargs` are forwarded verbatim to the native reader — match them to the
backend you chose.  When `backend="auto"`, a warning is logged naming the selected
backend if `read_kwargs` is non-empty, so you can pin it explicitly.

`required_columns(plan)` computes the projected column list explicitly when you need it:

```python
cols = required_columns(plan)  # schema cols + metric feature_names + profile_cols
```

**Statistical profiling columns** — to load columns beyond what metrics reference (e.g. for default statistics even without explicit `MetricSpec` entries), declare them on the plan:

```python
plan = MonitoringPlan(
    ...,
    profile_cols=["age", "income", "tenure"],  # loaded even if no metric targets them
)
```

### `ExcelSource` — Excel file on disk

`ExcelSource` reads a worksheet from an Excel file (`.xlsx`, `.xls`, and other formats
supported by the active backend).  Install the opt-in extra before use:

```bash
pip install ayn-ml[excel]
```

```python
from ayn_ml.data import ExcelSource

# auto backend — Polars (fastexcel) if available, else pandas (openpyxl)
source = ExcelSource(path="data/production.xlsx")
df_loaded = source.load(plan)   # projects to exactly what the plan needs
```

**Sheet selection** is a first-class parameter — pass a sheet name or a 0-based integer
index.  The integer convention follows pandas; the Polars backend translates automatically:

```python
source = ExcelSource(path="data/production.xlsx", sheet_name="run_42")
source = ExcelSource(path="data/production.xlsx", sheet_name=1)  # second sheet
```

**Explicit backend selection** lets you pick the engine and use its native kwargs directly:

```python
# Polars backend — use pl.read_excel kwargs (requires fastexcel)
source = ExcelSource(path="data/production.xlsx", backend="polars", read_kwargs={"infer_schema_length": 200})

# pandas backend — use pd.read_excel kwargs (requires openpyxl)
source = ExcelSource(path="data/production.xlsx", backend="pandas", read_kwargs={"usecols": ["age", "income"]})
```

`read_kwargs` are forwarded verbatim to the native reader — match them to the
backend you chose.  When `backend="auto"`, a warning is logged naming the selected
backend if `read_kwargs` is non-empty, so you can pin it explicitly.

## Window selection

A window strategy narrows a full DataFrame to the rows that represent the current monitoring period. Strategies are configured on the plan's `window` field so they round-trip through YAML.

| Config type | Strategy | When to use |
|---|---|---|
| `FullWindowConfig` | Pass through unchanged | DataFrame is already pre-filtered |
| `LastNRowsWindowConfig(n)` | Last N rows by position | Chronologically ordered DataFrame |
| `TimeWindowConfig(start, end)` | Filter by timestamp range | Explicit date boundaries |

```python
from datetime import datetime, timezone
from ayn_ml import MonitoringPlan, LastNRowsWindowConfig, TimeWindowConfig

# last 1000 rows
plan = MonitoringPlan(
    ...,
    window=LastNRowsWindowConfig(n=1_000),
)

# or with a time window
plan = MonitoringPlan(
    ...,
    window=TimeWindowConfig(
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 30, tzinfo=timezone.utc),
    ),
)
```

The strategy implementations can also be used directly at runtime:

```python
from ayn_ml.data.sampling import LastNRowsSampling, TimeWindowSampling

window = LastNRowsSampling(n=1_000).sample(df, schema)
window = TimeWindowSampling(start, end).sample(df, schema)
```

Both raise `InsufficientDataError` when the resulting window is empty.

## Random subsampling

`RandomSamplingConfig` reduces the current window size for performance, applied after window selection. Exactly one of `n` or `frac` must be provided.

```python
from ayn_ml import MonitoringPlan, RandomSamplingConfig

plan = MonitoringPlan(
    ...,
    sampling=RandomSamplingConfig(n=1_000, seed=42),   # absolute count
    # or:
    # sampling=RandomSamplingConfig(frac=0.1, seed=42), # 10% of the window
)
```

Can also be used directly:

```python
from ayn_ml.data.sampling import RandomSampling

subsampled = RandomSampling(n=1_000, seed=42).sample(window_df, schema)
```

Raises `InsufficientDataError` when the window is empty.

## Full pipeline examples

**Separate current and reference DataFrames** — pass each source independently; the reference is typically a training baseline or a frozen snapshot:

```python
from ayn_ml import MonitoringPlan, TabularSchema, MetricSpec
from ayn_ml.data.source import DataFrameSource

plan = MonitoringPlan(
    name="churn_monitor",
    model_id="churn_v2",
    model_version="1.0",
    data_schema=TabularSchema(),
    metrics=[MetricSpec(name="psi", feature_name="age")],
    profile_cols=["income", "tenure"],
)

# load each source projected to the columns the plan needs, then pass directly
df_current = DataFrameSource(raw_current).load(plan)
df_reference = DataFrameSource(raw_reference).load(plan)

# report = Runner().run(plan, current=df_current, reference=df_reference)
```

**Loading from CSV files:**

```python
from ayn_ml.data import CsvSource

df_current = CsvSource("data/current_week.csv").load(plan)
df_reference = CsvSource("data/training_baseline.csv").load(plan)

# report = Runner().run(plan, current=df_current, reference=df_reference)
```

**Loading from Excel files:**

```python
from ayn_ml.data import ExcelSource

df_current = ExcelSource("data/current_week.xlsx").load(plan)
df_reference = ExcelSource("data/training_baseline.xlsx", sheet_name="reference").load(plan)

# report = Runner().run(plan, current=df_current, reference=df_reference)
```

