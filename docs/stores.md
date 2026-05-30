# Stores

Stores are the persistence layer of ayn-ml. They implement two protocols:

| Protocol | Method | Role |
|---|---|---|
| `ResultSink` | `write(report)` | Write-only — notification channels |
| `ResultStore` | `write` + `read_history` + `get_report` | Read + write — full persistence |

Every `ResultStore` can also act as a `ResultSink`. Pass it to the Runner via
`store=` and it handles persistence automatically after each run.

## Choosing a store

| Store | Dependency | Use case |
|---|---|---|
| `InMemoryStore` | core | Tests, notebooks, quick exploration |
| `SqliteStore` | core (stdlib) | Local prod, CI, single-machine |
| `JsonStore` | core | *(coming soon)* |
| `ParquetStore` | `ayn-ml[parquet]` | *(coming soon)* |
| `SqlStore` | `ayn-ml[sql]` | *(coming soon)* Multi-machine, PostgreSQL/MySQL |
| `MlflowStore` | `ayn-ml[mlflow]` | *(coming soon)* |
| `S3Store` | `ayn-ml-pro[s3]` | Cloud S3-compatible storage — see below |
## InMemoryStore

Stores `MonitoringReport` objects in an in-memory `deque`. Thread-safe.
All data is lost when the process exits.

```python
from ayn_ml.stores import InMemoryStore

store = InMemoryStore()
report = Runner().run(plan, df, store=store)
rows = store.read_history("fraud_v3")
df   = pd.DataFrame(rows)
```

**`maxlen`** — keep only the N most recent reports. Useful for bounding
memory usage in long-running pipelines:

```python
store = InMemoryStore(maxlen=100)  # evicts oldest when full
```

## SqliteStore

Stores reports in a local SQLite file. No external deps (stdlib
`sqlite3`). Thread-safe via `threading.Lock`. Idempotent: a duplicate
`run_id` is silently ignored.

```python
from ayn_ml.stores import SqliteStore

# Standard usage
store = SqliteStore("monitoring.db")
report = Runner().run(plan, df, store=store)

# Context manager — close() called automatically
with SqliteStore("monitoring.db") as store:
    report = Runner().run(plan, df, store=store)

# In-memory db for tests (SQL behaviour, no file on disk)
store = SqliteStore(":memory:")
```

**Upgrading from an older version** — the schema has evolved between
releases. A db created before the `metric_type` column was added
(v0.x) will raise
`sqlite3.OperationalError: table metric_results has no column named metric_type`
on the first `write()`. Fix: recreate the db, or run manually:

```sql
ALTER TABLE metric_results ADD COLUMN metric_type TEXT;
```

**Cross-session persistence** — reopening the same file restores the full
history:

```python
store1 = SqliteStore("monitoring.db")
store1.write(report)
store1.close()

store2 = SqliteStore("monitoring.db")  # new conn
rows = store2.read_history("fraud_v3")  # retrieves all data
```
## `read_history()` — querying the history

Returns a list of flat dicts, newest first. Pass directly to `pd.DataFrame()`.

```python
rows = store.read_history(
    model_id="fraud_v3",         # required
    model_version="3.1",         # optional
    metric_name="roc_auc",       # optional
    metric_type="performance",   # optional — see below
    limit=50,                    # optional — None returns everything
    get_metadata=False,          # optional — see below
)
df = pd.DataFrame(rows)
```

**Columns always present:**

| Column | Description |
|---|---|
| `run_id` | Run id |
| `model_id` | Model id |
| `model_version` | Model version |
| `metric_name` | Metric name (or stat name for profile rows) |
| `feature_name` | Target column (drift, stats, profile) |
| `val` | Computed value |
| `status` | `True` / `False` / `None` (pass / fail / no threshold) |
| `effect_size` | Normalised effect size (when applicable) |
| `effect_size_label` | Effect size label |
| `period_start` | Start of the observation window (ISO 8601) |
| `period_end` | End of the observation window (ISO 8601) |
| `metric_type` | Row type — see next section |

### Filtering by `metric_type`

The `metric_type` column discriminates rows in `metric_results`:

| Value | Origin |
|---|---|
| `"performance"` | `MetricSpec` with `metric_type=MetricType.performance` |
| `"drift"` | `MetricSpec` with `metric_type=MetricType.drift` |
| `"statistics"` | `MetricSpec` with `metric_type=MetricType.stats` |
| `"fairness"` | `MetricSpec` with `metric_type=MetricType.fairness` |
| `None` | `MetricSpec` with no explicit `metric_type` |
| `"profile"` | Column profile stat (`enable_profiling=True`) |

```python
# Performance metrics only
perf = store.read_history("fraud_v3", metric_type="performance")

# Column profile stats only
profile = store.read_history("fraud_v3", metric_type="profile")
df_profile = pd.DataFrame(profile)
# df_profile.pivot(index="period_start", columns=["feature_name", "metric_name"], values="value")
```

Without `metric_type`, all rows are returned (metrics + profile).

### Enriching rows with `get_metadata=True`

Performs a JOIN with `monitoring_runs` and expands the plan JSON into
columns prefixed `plan_` / `run_`:

```python
rows = store.read_history("fraud_v3", get_metadata=True)
```

**Additional columns:**
`plan_name`, `plan_window_type`, `plan_window_n`,
`plan_sampling_type`, `plan_sampling_frac`, `run_n_current`, `run_n_reference`
## `get_report()` — retrieving a full report

```python
report = store.get_report(run_id)  # MonitoringReport | None
```

Reconstructs the complete `MonitoringReport` from the store:
`plan`, `ctx` (including `period_start` / `period_end`), `results`, `errors`,
`fired_alerts`, and `profile` (if `enable_profiling=True` was set at run time).

## Time-series pattern

```python
import pandas as pd
import matplotlib.pyplot as plt

rows = store.read_history("fraud_v3", metric_name="roc_auc")
df = pd.DataFrame(rows)
df["period_start"] = pd.to_datetime(df["period_start"])
df = df.sort_values("period_start")

plt.plot(df["period_start"], df["value"], marker="o")
plt.title("AUC over time — fraud_v3")
plt.show()
```

## Thread safety and idempotence

- **Thread-safe**: both stores use `threading.Lock` — concurrent calls from
  multiple threads carry no risk of data corruption.
- **Idempotent**: calling `write(report)` twice with the same `run_id` has
  no effect — the second call is silently ignored. Useful in pipelines with
  retry logic.

```python
store.write(report)  # ignored — run_id already present
assert len(store.read_history("m")) == len(original_results)
```

---

## S3Store — `ayn-ml-pro`

> **Status:** API and class structure are defined. `S3Store.save()`, `.load()`, and `.list_runs()` are not yet implemented.

**Import:** `from ayn_ml_pro.stores import S3Store`

> **Stub in `ayn-ml`:** `ayn_ml.stores` exports `S3Store: type[Any] | None = None`. This stub is populated by `ayn_ml_pro._extension` when `ayn-ml-pro` is installed; it is `None` otherwise. Accessing a `None` value raises `TypeError` at call time with a clear message. Import from `ayn_ml_pro.stores` directly in all production code.

**Extra required:** `pip install "ayn-ml-pro[s3]"`

`S3Store` implements the `ResultStore` protocol from `ayn-ml` (Apache 2.0). It serialises each `MonitoringReport` to JSON and writes it to any S3-compatible bucket (AWS S3, MinIO, GCS interop layer, etc.).

### Key format

Reports are stored under:

```
{prefix}{plan_name}/{run_id}.json
```

Example with `prefix="ayn-ml/"` and `plan_name="fraud_monitor"`:

```
ayn-ml/fraud_monitor/2024-06-01T12:00:00Z_abc123.json
```

### Constructor

```python
S3Store(
    bucket: str,                        # S3 bucket name
    prefix: str = "ayn-ml/",           # Key prefix for all reports
    region_name: str | None = None,     # AWS region
    endpoint_url: str | None = None,    # Custom endpoint (MinIO, GCS, etc.)
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
)
```

Credentials must not be hardcoded. Prefer IAM roles or env variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`):

```python
from ayn_ml_pro.stores import S3Store

# AWS — credentials from env / IAM role
store = S3Store(bucket="my-monitoring-bucket", prefix="ayn-ml/reports/", region_name="eu-west-1")

# MinIO or other S3-compatible store
store = S3Store(
    bucket="my-bucket",
    endpoint_url="http://minio:9000",
    aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
    aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
)

# Persist a report
store.write(report)

# Retrieve a report by run ID
report = store.load("abc123")            # -> MonitoringReport

# List all run IDs for a plan (oldest first)
run_ids = store.list_runs("fraud_monitor")  # -> list[str]
```

### Wiring to a Runner

```python
from ayn_ml.runner import Runner
from ayn_ml_pro.stores import S3Store

store = S3Store(bucket="my-monitoring-bucket", region_name="us-east-1")
runner = Runner(plan, store=store)
runner.run(df_current, ref=df_reference)
```

### Exception reference

| Exception | When raised |
|---|---|
| `StoreConnectionError` | S3 write, read, or list call failed |
| `KeyError` | `load()` called with a run ID that does not exist in the bucket |
| `ImportError` | `boto3` not installed — install with `pip install "ayn-ml-pro[s3]"` |
