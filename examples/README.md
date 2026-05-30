# Examples

Interactive notebooks for exploring ayn-ml. Each notebook is self-contained and generates its own synthetic data.

## Setup — VS Code (recommended)

**Prerequisites:** [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) + [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

### With uv

```bash
# install uv once (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repo root
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e "." ipykernel
```

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e "." ipykernel
```

Then open any `.ipynb` file in VS Code, click **Select Kernel** (top-right), choose **Python Environments**, and pick `.venv`.

---

## Setup — classic Jupyter / JupyterLab

### With uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e "." ipykernel jupyter notebook
python -m ipykernel install --user --name ayn-ml --display-name "ayn-ml"
jupyter notebook examples/
```

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "." ipykernel jupyter notebook
python -m ipykernel install --user --name ayn-ml --display-name "ayn-ml"
jupyter notebook examples/
```

### With Poetry

```bash
poetry install
poetry add --group dev jupyter notebook ipykernel
poetry run python -m ipykernel install --user --name ayn-ml --display-name "ayn-ml"
poetry run jupyter notebook examples/
```

---

## Notebooks

| Notebook | What it covers |
|---|---|
| [01_core_and_data_layer.ipynb](01_core_and_data_layer.ipynb) | MonitoringPlan, schemas, MetricSpec, YAML round-trip, DataFrameSource, sampling, partitioning |
| [02_metrics.ipynb](02_metrics.ipynb) | Registry, metric resolution, computing tabular metrics directly on synthetic data |
| [03_cbpe.ipynb](03_cbpe.ipynb) | CBPE — performance estimation without ground-truth labels |
| [04_fairness.ipynb](04_fairness.ipynb) | Fairness metrics: demographic parity, equalized odds, disparate impact |
| [05_recsys_metrics.ipynb](05_recsys_metrics.ipynb) | Recsys metrics — RecSysSchema, interactions_from_matrix, ranking accuracy (precision, recall, NDCG, MAP, MRR), beyond-accuracy (diversity, novelty, popularity bias, personalization, item/user bias) |
| [06_runner_and_profiling.ipynb](06_runner_and_profiling.ipynb) | Runner — strict/lenient mode, parallel execution, statistical profiling |
| [07_stores.ipynb](07_stores.ipynb) | Stores — InMemoryStore and SqliteStore, read_history, get_report, time-series plots |
| [08_renderers_and_alerts.ipynb](08_renderers_and_alerts.ipynb) | Renderers — HtmlRenderer snapshot and history dashboard; AlertRule, ThresholdPolicy, WebhookChannel, EmailChannel |
| [09_advisor.ipynb](09_advisor.ipynb) | MetricAdvisor — automatic MonitoringPlan generation from data characteristics; imbalance routing, Levene routing, end-to-end with Runner |
