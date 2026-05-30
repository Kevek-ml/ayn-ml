# Runner — CloudRunner Scheduling

> **Status:** API and class structure are defined. `run_once()`, `start()`, `stop()`, and `audit_log()` are not yet implemented.

## CloudRunner

**Import:** `from ayn_ml_pro.runner import CloudRunner`

`CloudRunner` extends the synchronous `Runner` from `ayn-ml` (Apache 2.0) with scheduling, parallelism, retry logic, and an immutable audit trail. It is the recommended execution layer for production deployments.

### Feature overview

| Feature | Description |
|---------|-------------|
| **Scheduling** | Cron expressions (`"0 * * * *"`) or shorthands (`@hourly`, `@daily`) |
| **Parallelism** | Concurrent metric computation across feature columns (`max_workers`) |
| **Retry logic** | Configurable exponential back-off on transient failures |
| **Audit trail** | Append-only log of every run: ID, timestamps, status, errors |

### Constructor

```python
CloudRunner(
    plan: MonitoringPlan,
    store: ResultStore,
    sinks: list[ResultSink] | None = None,
    schedule: str | None = None,        # Cron expr or @shorthand
    max_workers: int = 4,               # Parallel workers
    max_retries: int = 3,               # Max retry attempts per run
    retry_backoff: float = 2.0,         # Base back-off in seconds (exponential)
)
```

### Schedule syntax

| Expression | Meaning |
|------------|---------|
| `"0 * * * *"` | Every hour |
| `"0 8 * * *"` | Daily at 08:00 |
| `"@hourly"` | Shorthand for every hour |
| `"@daily"` | Shorthand for daily at midnight |
| `None` | No automatic scheduling — use `run_once()` manually |

### Usage

#### Scheduled execution

```python
from ayn_ml_pro.runner import CloudRunner
from ayn_ml_pro.stores import S3Store
from ayn_ml_pro.sinks import SlackChannel
import os

store = S3Store(bucket="my-bucket", region_name="eu-west-1")
sink = SlackChannel(token=os.environ["SLACK_BOT_TOKEN"], channel="#ml-alerts")

runner = CloudRunner(
    plan,
    store=store,
    sinks=[sink],
    schedule="0 * * * *",   # hourly
    max_workers=8,
    max_retries=3,
    retry_backoff=2.0,
)

runner.start()   # blocks until runner.stop() is called
```

#### One-shot execution

```python
runner = CloudRunner(plan, store=store)
report = runner.run_once(df_current, ref=df_reference)
print(report.summary)
```

#### Stopping the runner

```python
import signal

def handle_sigterm(sig, frame):
    runner.stop()   # waits for any in-flight run to complete

signal.signal(signal.SIGTERM, handle_sigterm)
runner.start()
```

### Audit trail

`audit_log()` returns an immutable list of run records. Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Unique identifier for the run |
| `started_at` | `str` | ISO 8601 timestamp |
| `finished_at` | `str` | ISO 8601 timestamp |
| `status` | `str` | `"success"` or `"error"` |
| `err` | `str \| None` | Error message if `status == "error"` |

```python
for record in runner.audit_log():
    print(record["run_id"], record["status"], record.get("err"))
```

### Error handling

`CloudRunner` catches per-metric exceptions and stores them in `MonitoringReport.errors` — a single metric failure never aborts the run. If a run fails after all retries, `CloudRunnerError` is raised and recorded in the audit trail.

---

## Exception reference

| Exception | When raised |
|-----------|-------------|
| `CloudRunnerError` | Run fails after all retry attempts, or the scheduler cannot start |
