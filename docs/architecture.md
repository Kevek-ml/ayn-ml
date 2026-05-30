# ayn-ml — Architecture

Canonical reference for module structure, type contracts, implementation specs, and design decisions.
Read this in full before implementing any new phase.

---

## Module structure (target)

```
ayn_ml/
├── core/
│   ├── schema.py        # BaseSchema, TabularSchema, TextSchema, AgentSchema, RecSysSchema  ✅
│   ├── spec.py          # MetricSpec, MonitoringPlan                          ✅
│   ├── result.py        # ExecutionContext, MetricResult, MetricError, FiredAlert, MonitoringReport  ✅
│   ├── alert.py         # AlertPolicy (Protocol), ThresholdPolicy, AlertRule   ✅
│   └── data_selection.py   # WindowConfig, RandomSamplingConfig, PartitioningConfig       ✅
├── metrics/
│   ├── registry.py      # @register_metric, get_metric, list_metrics           ✅
│   ├── base.py          # Metric Protocol + compute_status                     ✅
│   ├── recsys.py        # precision_at_k, recall_at_k, fbeta_at_k, hit_rate, map_at_k, ndcg_at_k, mrr_at_k, diversity, novelty, popularity_bias, personalization, item_bias, user_bias, serendipity (14 metrics)  ✅
│   ├── recsys_utils.py  # interactions_from_matrix — convert user×item matrix to long interactions table  ✅
│   └── tabular/
│       ├── _helpers.py      # _result_metric, _result_stat, extract_feature, _require_reference, to_float_array, shared extractors  ✅
│       ├── performance.py   # accuracy → mape (12 metrics)                     ✅
│       ├── drift.py         # psi, wasserstein, mmd, target_drift              ✅
│       ├── tests.py         # ks_2samp → ztest_proportions (15 metrics)        ✅
│       ├── statistics.py    # mean → empty_columns (18 metrics)               ✅
│       ├── estimation.py    # cbpe_* (5 metrics)                               ✅
│       ├── fairness.py      # demographic_parity, equalized_odds, disparate_impact ✅
│       └── profiler.py      # profile_columns() — standalone, not in registry   ✅
│   ├── nlp/                 # Phase 8
│   │   ├── quality.py       # BLEUMetric, ROUGEMetric, BERTScoreMetric
│   │   ├── drift.py         # EmbeddingDriftMetric (MMD dans espace embeddings)
│   │   └── safety.py        # ToxicityMetric, PIIMetric, HallucinationMetric
│   └── agent/               # Phase 9
│       ├── performance.py   # TaskCompletionMetric, ToolSuccessMetric
│       └── cost.py          # AvgTokensMetric, AvgLatencyMetric, CostMetric
├── data/
│   ├── source.py        # DataSource (ABC) + DataFrameSource  ✅  (ParquetSource, SqlSource — Phase 5)
│   ├── csv.py           # CsvSource  ✅
│   ├── excel.py         # ExcelSource  ✅  (opt-in: pip install ayn-ml[excel])
│   ├── sampling.py      # SamplingStrategy + FullData, LastN, TimeWindow, Random       ✅
│   └── partitioner.py   # DataPartitioner + FixedReference, TimeBased                  ✅
├── models/              # Phase 5
│   ├── base.py          # ModelWrapper Protocol
│   ├── sklearn.py       # SklearnWrapper
│   ├── huggingface.py   # HuggingFaceWrapper
│   ├── langchain.py     # LangChainWrapper
│   └── mlflow_loader.py # MlflowModelLoader (opt-in)
├── runner.py            # Runner (stateless) ✅
├── renderers/           # Phase 6b ✅
│   ├── base.py          # ReportRenderer Protocol, ChartBackend Protocol  ✅
│   ├── html.py          # HtmlRenderer (core, jinja2)                     ✅
│   ├── plotly.py        # PlotlyBackend (core)                            ✅
│   └── no_chart.py      # NoChartBackend (headless)                       ✅
├── stores/              # Phase 6a ✅ (InMemoryStore + SqliteStore)
│   ├── base.py          # ResultStore Protocol                         ✅
│   ├── _helpers.py      # to_row(), extract_plan_meta(), report_to_rows() ✅
│   ├── memory.py        # InMemoryStore (core) — tests et notebooks    ✅
│   ├── sqlite.py        # SqliteStore (core, zero dep)                 ✅
│   ├── json_store.py    # JsonStore (core) — Phase 7 suite
│   ├── parquet.py       # ParquetStore (opt-in) — Phase 7 suite
│   ├── sql.py           # SqlStore (opt-in, SQLAlchemy) — Phase 7 suite
│   └── mlflow.py        # MlflowStore (opt-in) — Phase 7 suite
├── sinks/               # Phase 6b ✅ (ResultSink protocol + channels)
│   ├── base.py          # ResultSink Protocol                          ✅
│   ├── email.py         # EmailChannel (core, smtplib)                 ✅
│   └── webhook.py       # WebhookChannel (core, urllib)                ✅
├── advisor/             # ✅ automatic MonitoringPlan generation from data
│   ├── __init__.py      # exports: MetricAdvisor, SuggestedPlan       ✅
│   ├── advisor.py      # MetricAdvisor — public API                  ✅
│   ├── _plan.py         # SuggestedPlan — output dataclass             ✅
│   ├── _analysis.py     # analyze_columns(), ColumnAnalysis dataclass  ✅
│   └── _rules.py        # suggest_drift_specs(), suggest_performance_specs(), suggest_statistics_specs()  ✅
├── explain/             # Phase 10–11
│   ├── drift_impact.py  # DriftAttributor — Apache 2.0
│   ├── shap_monitor.py  # SHAP distribution monitoring — commercial edition
│   ├── llm_judge.py     # LLM-as-judge — commercial edition
│   └── experimental/    # commercial edition
├── io/
│   └── plan_parser.py   # MonitoringPlan.from_yaml / to_yaml  (stub — Phase 6)
└── exceptions.py        # Hiérarchie d'exceptions              ✅
```

---

## Types core — décisions figées

### MetricType

```python
class MetricType(str, Enum):
    performance = "performance"
    drift       = "drift"
    statistics  = "statistics"
    fairness    = "fairness"
    nlp_quality = "nlp_quality"
    nlp_drift   = "nlp_drift"
    nlp_safety  = "nlp_safety"
    agent_performance = "agent_performance"
    agent_cost  = "agent_cost"
    recsys      = "recsys"
    custom      = "custom"
```

### MetricSpec (frozen)

Ce qu'on calcule. Agnostique du modèle et du temps — réutilisable sur plusieurs versions.
`model_id` et `model_version` sont dans `ExecutionContext`, PAS ici.
Raison : la même spec doit s'appliquer à plusieurs versions (champion/challenger, A/B).

```python
class MetricSpec(BaseModel):
    name: str                                       # clé dans le registry
    metric_type: MetricType | None = None           # None = le registry détermine le type
    feature_name: str | None = None                 # None = métrique globale (ex: accuracy)
    params: dict[str, Any] = Field(default_factory=dict)
    threshold: float | list[float] | None = None
    upper_bound: bool = True                        # True = valeur doit être <= threshold
    model_config = ConfigDict(frozen=True)
```

### ExecutionContext (frozen)

Tout ce qui varie d'un run à l'autre. Construit par le Runner.

```python
class ExecutionContext(BaseModel):
    run_id: str                            # UUID hex auto-généré — corrèle les lignes dans un store
    model_id: str
    model_version: str
    eval_timestamp: datetime
    period_start: datetime | None = None   # None si timestamp_col non configuré
    period_end: datetime | None = None
    n_current: int | None = None           # lignes dans la fenêtre current après filtrage
    n_reference: int | None = None         # lignes dans la fenêtre reference, None si absente
    model_config = ConfigDict(frozen=True)
```

### MetricResult

Résultat de calcul pur. Porte spec + valeur + statut.
Pas de context — celui-ci vit dans `MonitoringReport` pour éviter la duplication.

```python
class MetricResult(BaseModel):
    spec: MetricSpec
    value: float | int | str | None = None
    status: bool | None = None                      # None si pas de threshold
    conf_interval: tuple[float, float] | None = None
    effect_size: float | None = None                # Cohen's d, Cliff's delta, KS D…
    effect_size_label: str | None = None            # "cohen_d" | "cliff_delta" | "ks_statistic"…
```

### ColumnType

Resolved classification of a DataFrame column, produced once per run by `classify_columns()` in `ayn_ml.metrics.tabular._helpers` and used by the Runner to route metrics before calling `compute()`.

```python
class ColumnType(str, Enum):
    numeric     = "numeric"      # continuous or ordinal (float or multi-valued int)
    categorical = "categorical"  # string or integer-encoded categorical
    binary      = "binary"       # int column whose unique values ⊆ {0, 1}
```

Resolution order per column:
1. `TabularSchema.feature_types` explicit declaration — wins.
2. numpy dtype inference: integer dtype with unique values `⊆ {0, 1}` → `binary`; other numeric → `numeric`; everything else → `categorical`.

`binary` is intentionally distinct from `numeric` and `categorical`: it is accepted by both continuous tests (KS, Wasserstein) and the chi-square homogeneity test.

---

### DataSchema — hiérarchie par modalité

```python
class BaseSchema(BaseModel):
    # Tous à None par défaut — déclarer uniquement les colonnes présentes dans le DataFrame.
    # Le Runner valide leur présence au runtime (strict=True par défaut).
    timestamp_col: str | None = None
    model_id_col: str | None = None
    model_version_col: str | None = None

class TabularSchema(BaseSchema):
    type: Literal["tabular"] = "tabular"            # discriminator Pydantic — obligatoire
    label_col: str = "y_true"
    prediction_col: str = "y_pred"
    proba_col: str | None = "y_pred_proba"
    feature_types: dict[str, Literal["numeric", "categorical"]] = {}
    protected_cols: list[str] | None = None         # colonnes d'attributs protégés (fairness)

class TextSchema(BaseSchema):
    type: Literal["text"] = "text"                  # discriminator Pydantic — obligatoire
    input_col: str = "input_text"
    output_col: str = "output_text"
    reference_col: str | None = "reference_text"
    embedding_col: str | None = None

class AgentSchema(BaseSchema):
    type: Literal["agent"] = "agent"                # discriminator Pydantic — obligatoire
    input_col: str = "input"
    output_col: str = "output"
    trace_col: str = "trace"
    success_col: str | None = "success"
    tool_calls_col: str | None = "tool_calls"
    tokens_used_col: str | None = "tokens_used"
    latency_col: str | None = "latency_ms"
    cost_col: str | None = "cost_usd"

DataSchema = Annotated[
    TabularSchema | TextSchema | AgentSchema | RecSysSchema,
    Field(discriminator="type"),
]
```

### WindowConfig, RandomSamplingConfig, PartitioningConfig (`core/data_selection.py`)

Configuration déclarative pour la sélection des données. `WindowConfig` et `RandomSamplingConfig` vivent dans `MonitoringPlan` (round-trippable JSON/YAML). `PartitioningConfig` est un alias de type pour `TimeBasedPartitioningConfig` — instancié directement à l'exécution, pas déclaré sur le plan.

```python
# WindowConfig — sélection de la fenêtre courante
class FullWindowConfig(BaseModel):
    type: Literal["full"] = "full"          # identité — DataFrame utilisé tel quel

class LastNRowsWindowConfig(BaseModel):
    type: Literal["last_n"] = "last_n"
    n: int                                  # N dernières lignes (ordre chronologique supposé)

class TimeWindowConfig(BaseModel):
    type: Literal["time_window"] = "time_window"
    start: datetime
    end: datetime                           # [start, end] inclus

WindowConfig = Annotated[
    FullWindowConfig | LastNRowsWindowConfig | TimeWindowConfig,
    Field(discriminator="type"),
]

# RandomSamplingConfig — sous-échantillonnage pour la performance
class RandomSamplingConfig(BaseModel):
    type: Literal["random"] = "random"
    n: int | None = None                    # exactement un de n ou frac requis
    frac: float | None = None              # frac dans (0, 1]
    seed: int | None = None

# PartitioningConfig — split current / reference depuis une seule source
class TimeBasedPartitioningConfig(BaseModel):
    type: Literal["time_based"] = "time_based"
    cutoff: datetime
    reference_window: timedelta
    # current   → timestamp > cutoff
    # reference → timestamp dans (cutoff - reference_window, cutoff]

PartitioningConfig = TimeBasedPartitioningConfig
```

### MonitoringPlan

Config déclarative et sérialisable (YAML/JSON). Une instance par modèle.

```python
class MonitoringPlan(BaseModel):
    name: str
    model_id: str
    model_version: str
    data_schema: DataSchema
    metrics: list[MetricSpec]
    description: str = ""
    enable_profiling: bool = False     # when True, Runner profiles watched cols and attaches to MonitoringReport.profile
    window: WindowConfig | None = None
    sampling: RandomSamplingConfig | None = None
```

### FiredAlert

Enregistrement léger d'une alerte déclenchée. Distinct de `AlertRule` (config) — porte seulement ce qui est nécessaire pour logguer ou notifier.

```python
@dataclass
class FiredAlert:
    metric_name: str          # nom de la métrique qui a déclenché l'alerte
    policy_type: str          # "threshold" | "change" | "consecutive"
    details: dict[str, Any]   # contexte spécifique à la policy
```

### MonitoringReport

Sortie du Runner. Résultats + erreurs non-fatales (une métrique qui plante n'arrête pas tout).

```python
@dataclass
class MonitoringReport:
    plan: MonitoringPlan
    context: ExecutionContext
    results: list[MetricResult]
    errors: list[MetricError]
    fired_alerts: list[FiredAlert] = field(default_factory=list)
    profile: dict[str, dict[str, float | int | str | None]] | None = None
    # profile is set by the Runner when plan.enable_profiling=True; None otherwise.
    # Keys are column names; values are ColumnProfile dicts (see profiler.py).

    def to_dict(self) -> dict: ...
    # Includes "profile" key only when profile is not None.
    def to_dataframe(self) -> Any: ...   # tidy DataFrame: une ligne par MetricResult + champs de contexte
```

---

## Metric Protocol + Registry

```python
class Metric(Protocol):
    name: str
    metric_type: MetricType
    requires_reference: bool   # True pour drift et tests statistiques; False pour performance et statistiques

    # Column-type routing — checked by the Runner before compute() is called.
    # Omitting accepted_column_types skips the feature-column check entirely.
    accepted_column_types: frozenset[ColumnType]                  # which ColumnType values are valid for spec.feature_name
    accepted_target_types: dict[str, frozenset[ColumnType]]       # schema attr → accepted ColumnType set (fairness metrics only)

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult: ...
```

---

## Statistical Profiling — `metrics/tabular/profiler.py`

Standalone module. Not part of the metric registry and not configurable via `MetricSpec`.
Activated by setting `MonitoringPlan.enable_profiling = True`; the Runner calls it after metric execution.

```python
ColumnProfile = dict[str, float | int | str | None]

def profile_columns(
    df: Any,
    col_names: list[str],
    schema: DataSchema,
) -> dict[str, ColumnProfile]:
    """Compute a statistical profile for each requested column.

    Uses classify_columns() to resolve each column's ColumnType (respecting
    TabularSchema.feature_types overrides).  Columns absent from df are
    silently skipped.

    Numeric / binary columns → min, max, mean, std, p25, p50, p75, null_count, null_pct
    Categorical columns      → null_count, null_pct, n_unique, top_category
    """
```

**Columns profiled by the Runner (when `enable_profiling=True`):**
- Every `spec.feature_name` declared across all `MetricSpec` entries in the plan
- All columns declared on the schema (`schema.column_names`) — covers `prediction_col`, `label_col`, `proba_col`, `timestamp_col`, and any other schema-level columns

The result is a `dict[str, ColumnProfile]` attached to `MonitoringReport.profile` and included in `to_dict()` under the `"profile"` key.

---

**`accepted_column_types`** — declared on metrics that operate on `spec.feature_name`. The Runner checks `column_types[spec.feature_name]` against this frozenset before calling `compute()`. If the resolved `ColumnType` is not in the set, a `MetricError(error_type="SchemaError")` is returned without calling `compute()`. Attribute absent → check skipped (no routing).

**`accepted_target_types`** — declared on metrics that also validate schema-level columns such as `prediction_col` or `label_col` (currently used only by the three fairness metrics). Keys are `TabularSchema` attribute names (e.g. `"prediction_col"`); values are the accepted `ColumnType` frozensets for that column. Same early-return semantics as `accepted_column_types`.

Ajouter une métrique = une classe + `@register_metric("nom")`. Aucune modification de fichiers existants.

**Règle d'implémentation :** preprocessing avec narwhals, computation statistique via `.to_numpy()` (scipy/sklearn). Ne pas réimplémenter scipy en narwhals — la conversion est négligeable (zero-copy avec Polars/Arrow).

---

## Data Layer

### DataSource

```python
# Implémenté ✅
class DataSource(ABC):
    def load(self, plan: MonitoringPlan) -> Any: ...
    # Projette automatiquement aux colonnes requises par le plan

class DataFrameSource(DataSource):
    def __init__(self, data: pd.DataFrame | pl.DataFrame): ...

class CsvSource(DataSource):            # ✅ — any narwhals eager backend (auto: Polars → pandas)
    path: str | Path
    backend: str = "auto"               # "polars" | "pandas" | "modin" | "cudf" | "pyarrow"
    separator: str = ","                # normalised by narwhals across all backends
    read_kwargs: dict[str, Any] = {}    # forwarded verbatim to the native reader

class ExcelSource(DataSource):          # ✅ — pip install ayn-ml[excel]
    path: str | Path
    backend: str = "auto"               # "polars" (fastexcel) | "pandas" (openpyxl) | "auto"
    sheet_name: str | int = 0           # name or 0-based index (pandas convention)
    read_kwargs: dict[str, Any] = {}    # forwarded verbatim to the native reader

# À implémenter — Phase 5
class ParquetSource(DataSource):
    path: str
    columns: list[str] | None = None
    filters: list | None = None         # pushdown filtering pyarrow

class SqlSource(DataSource):            # pip install ayn-ml[sql]
    connection_string: str              # SQLAlchemy URI
    query: str
```

### SamplingStrategy

```python
class SamplingStrategy(ABC):
    def sample(self, df: Any, schema: DataSchema) -> Any: ...

class FullDataSampling(SamplingStrategy):
    pass  # identité — retourne le DataFrame tel quel

class LastNRowsSampling(SamplingStrategy):
    def __init__(self, n: int): ...          # n doit être positif

class TimeWindowSampling(SamplingStrategy):
    def __init__(self, start: datetime, end: datetime): ...  # filtre [start, end] inclus

class RandomSampling(SamplingStrategy):
    def __init__(self, n: int | None = None, frac: float | None = None, seed: int | None = None): ...
    # exactement un de n ou frac requis
```

### DataPartitioner

```python
class DataPartitioner(ABC):
    def partition(self, df: Any, schema: DataSchema) -> tuple[Any, Any | None]: ...

class FixedReferencePartitioner(DataPartitioner):
    def __init__(self, reference: Any): ...

class TimeBasedPartitioner(DataPartitioner):
    def __init__(self, cutoff: datetime, reference_window: timedelta): ...
    # current   → rows avec timestamp > cutoff
    # reference → rows avec timestamp dans (cutoff - reference_window, cutoff]
```

---

## Model Layer

```python
class ModelWrapper(Protocol):
    def predict(self, data: Any, schema: DataSchema) -> Any: ...

class SklearnWrapper:
    def __init__(self, model: Any, predict_proba: bool = False): ...

class HuggingFaceWrapper:
    def __init__(self, pipeline: Any): ...

class LangChainWrapper:
    def __init__(self, chain: Any): ...

# pip install ayn-ml[mlflow]
class MlflowModelLoader:
    def load(self, model_uri: str) -> ModelWrapper: ...
    # model_uri = "models:/churn_model/Production"
```

---

## Runner

Stateless. `ExecutionContext` construit depuis le plan + données.

```python
class Runner:
    def __init__(self, n_jobs: int = 1, strict: bool = True): ...
    # n_jobs=1 séquentiel, n_jobs=-1 tous les CPUs, n_jobs>=2 pool de threads
    # strict=True (défaut) : SchemaError si colonne déclarée absente du DataFrame
    # strict=False : warning + dégradation gracieuse — pour l'évaluation exploratoire

    def run(
        self,
        plan: MonitoringPlan,
        current: pd.DataFrame | pl.DataFrame | DataSource,
        reference: pd.DataFrame | pl.DataFrame | DataSource | None = None,
        model: ModelWrapper | None = None,
        store: ResultStore | None = None,
        sinks: list[ResultSink] | None = None,
        alert_rules: list[AlertRule] | None = None,
    ) -> MonitoringReport:
        # 1. Résout les DataSources (wrap DataFrames bruts si nécessaire)
        # 2. Applique la fenêtre (plan.window)
        # 3. Sous-échantillonnage aléatoire (plan.sampling, en plus de la fenêtre)
        # 4. Filtre les lignes par model_id + model_version (via schema)
        # 5. Valide les colonnes (_validate_plan_columns) — SchemaError si strict=True
        #    · time_window sans timestamp_col configuré → toujours SchemaError
        #    · colonnes d'infrastructure absentes → SchemaError ou warning selon strict
        #    · feature_name absents → SchemaError ou warning selon strict
        # 6. Construit ExecutionContext (run_id, bornes de période, n_current, n_reference)
        # 7. Classifie les colonnes une seule fois via classify_columns(current_df, schema)
        #    → dict[str, ColumnType] réutilisé pour chaque métrique du plan
        #    · ColumnType routing : si metric.accepted_column_types est défini et que
        #      ColumnType de spec.feature_name n'y figure pas → MetricError(SchemaError)
        #      sans appel à compute()
        #    · accepted_target_types : idem pour les colonnes schema (prediction_col, label_col…)
        # 8. Exécute chaque MetricSpec via le registry — erreurs isolées par métrique
        # 9. Évalue les alert_rules (ThresholdPolicy — stateless, pas de store requis)
        #     · FiredAlert créé par rule si policy.evaluate(result) → True
        # 10. Profiling statistique (opt-in via plan.enable_profiling) :
        #     · Collecte les feature_name de chaque spec + toutes les colonnes déclarées
        #       sur le schema via schema.column_names
        #     · Appelle profile_columns(current_df, sorted(feature_cols), plan.data_schema)
        #     · Attache le résultat à MonitoringReport.profile
        # 11. Écrit le report dans store (toujours, si fourni)
        # 12. Dispatch sinks inconditionnels (sinks= — chaque run, peu importe les alertes)
        # 13. Dispatch channels des AlertRule ayant firé (report complet, fired_alerts non vide)
        # Note : model.predict() → Phase 5 (model= accepté mais ignoré avec warning)
```

A scheduled cloud runner with parallelism, retry logic, and audit trail is available in the commercial edition.

**Règle store/sinks :**
- `store` reçoit toujours le report (persistance)
- `sinks` reçoivent le report à chaque run, inconditionnellement (ex : sink de logging, push dashboard)
- Les canaux de notification liés aux `AlertRule` reçoivent le report uniquement quand l'alerte se déclenche
- `ResultStore` étend `ResultSink` mais ne doit pas figurer dans `sinks`

---

## Alertes

### AlertPolicy

```python
@runtime_checkable
class AlertPolicy(Protocol):
    """Stateless callable protocol: evaluate(result) -> bool."""
    @property
    def policy_type(self) -> str: ...
    def evaluate(self, result: MetricResult) -> bool: ...
    def details(self, result: MetricResult) -> dict[str, Any]: ...

@dataclass(frozen=True)
class ThresholdPolicy:
    """Fires when MetricResult.status is False, or when value crosses a custom threshold."""
    threshold: float | None = None   # None = delegate to result.status
    upper_bound: bool = True         # True → fire when value > threshold; False → value < threshold
```

**Implémenté ✅ :** `ThresholdPolicy` — stateless, pas de store requis.

**Déféré (v2) :**
- `ChangePolicy(pct_change: float)` — alerte si variation > X% vs run précédent (nécessite `read_history()`)
- `ConsecutivePolicy(n: int)` — alerte si `status == False` pendant N runs consécutifs (nécessite `read_history()`)

### AlertRule

```python
@dataclass
class AlertRule:
    metric_name: str          # doit correspondre à un MetricSpec.name dans le plan
    policy: AlertPolicy
    channels: list[ResultSink] = field(default_factory=list)
```

| Channel | Extra requis | Usage |
|---------|-------------|-------|
| `EmailChannel` | aucun (smtplib) | Alertes email |
| `WebhookChannel` | aucun (urllib) | POST JSON vers n'importe quel endpoint — accepte `extra_headers` et `timeout` |

---

## Stores et Sinks

### ResultSink (write-only) ✅

```python
class ResultSink(Protocol):
    def write(self, report: MonitoringReport) -> None: ...
```

Implémenté dans `ayn_ml/sinks/base.py`. `EmailChannel` et `WebhookChannel` implémentent ce protocole.

### ResultStore (bidirectionnel) ✅

`ResultStore` étend `ResultSink` — tout store peut aussi servir de sink.

```python
class ResultStore(ResultSink, Protocol):
    def write(self, report: MonitoringReport) -> None: ...

    def read_history(
        self,
        model_id: str,
        model_version: str | None = None,
        metric_name: str | None = None,
        limit: int | None = None,        # None = pas de limite
        get_metadata: bool = False,
    ) -> list[dict[str, Any]]: ...       # flat rows, newest first

    def get_report(self, run_id: str) -> MonitoringReport | None: ...
```

**`read_history()` — colonnes retournées (sans `get_metadata`) :**
`run_id`, `model_id`, `model_version`, `metric_name`, `feature_name`, `value`, `status`,
`effect_size`, `effect_size_label`, `period_start`, `period_end`

**Avec `get_metadata=True`** — colonnes additionnelles (JOIN avec `monitoring_runs`, plan JSON éclaté) :
`plan_name`, `plan_window_type`, `plan_window_n`, `plan_sampling_type`, `plan_sampling_frac`,
`run_n_current`, `run_n_reference`

Le retour est directement `pd.DataFrame()`-compatible.

### Stores implémentés ✅

| Store | Dépendance | Statut | Notes |
|-------|-----------|--------|-------|
| `InMemoryStore` | core | ✅ | `collections.deque`, `maxlen` optionnel, thread-safe — idéal tests et notebooks |
| `SqliteStore` | core (stdlib) | ✅ | zero dep, 4 tables normalisées, idempotent sur `run_id` |
| `JsonStore` | core | À faire | fichier JSON local, zero dependency |
| `ParquetStore` | `ayn-ml[parquet]` | À faire | efficace pour l'historique |
| `SqlStore` | `ayn-ml[sql]` | À faire | SQLAlchemy, PostgreSQL/MySQL/SQLite |
| `MlflowStore` | `ayn-ml[mlflow]` | À faire | JSON artifact + `log_metric()` par métrique pour l'UI |

### SqliteStore — schéma SQLite ✅

Quatre tables normalisées, ouvertes avec `check_same_thread=False` + `threading.Lock`.

```
monitoring_runs          — une ligne par exécution Runner
  run_id TEXT PK
  model_id, model_version, eval_timestamp
  n_current, n_reference
  plan_json TEXT         — MonitoringPlan complet sérialisé

metric_results           — une ligne par MetricResult
  id INTEGER PK AUTOINCREMENT
  run_id FK
  model_id, model_version          — dénormalisés (pas de JOIN pour les time-series)
  metric_name, feature_name
  value, status, effect_size, effect_size_label
  period_start, period_end         — dénormalisés (requêtes temporelles sans JOIN)

metric_errors            — une ligne par MetricError
  id, run_id FK, metric_name, error_type, message

fired_alerts             — une ligne par FiredAlert
  id, run_id FK, metric_name, policy_type, details_json
```

Index : `(model_id, model_version)` et `(model_id, metric_name, period_start)`.

`write()` est idempotent — un doublon de `run_id` est détecté et ignoré silencieusement.

### Usage des stores avec le Runner

```python
from ayn_ml.stores import InMemoryStore, SqliteStore

# In-memory (tests, notebooks)
store = InMemoryStore()
report = Runner().run(plan, df, store=store)
rows = store.read_history("fraud_v2", metric_name="auc")
df_history = pd.DataFrame(rows)

# SQLite persistant (prod locale, CI)
with SqliteStore("monitoring.db") as store:
    report = Runner().run(plan, df, store=store)
    rows = store.read_history("fraud_v2", get_metadata=True)
```

| Modalité | Store recommandé | Sink d'alerte |
|----------|-----------------|---------------|
| Dev / CI | InMemoryStore | — |
| Prod locale (1 machine) | SqliteStore | EmailChannel, WebhookChannel |
| Tabular ML (multi-machine) | SqlStore, MlflowStore | EmailChannel, WebhookChannel |
| NLP / LLM | JsonStore, ParquetStore | WebhookChannel |
| Agent | JsonStore, MlflowStore | WebhookChannel |

---

## Advisor — `ayn_ml/advisor/`

Automatic `MonitoringPlan` generation from data characteristics.  Not a runtime component — call it once during setup to bootstrap a plan, then pass the resulting plan to `Runner`.

### MetricAdvisor

```python
class MetricAdvisor:
    def __init__(self, schema: TabularSchema) -> None: ...

    def suggest(
        self,
        df: Any,
        *,
        reference: Any,
        task_type: str = "classification",  # "classification" | "regression"
        name: str = "suggested_plan",
        model_id: str = "",
        model_version: str = "",
    ) -> SuggestedPlan: ...
```

The constructor accepts a `TabularSchema` once so the same instance can be called with different DataFrames (e.g. different time windows) without re-specifying the schema.

`suggest()` analyses each feature column (normality, skewness, sample size, optional variance ratio vs. `reference`) and assembles a `MonitoringPlan` containing:

- **Performance specs** — selected by `task_type` and class imbalance ratio.
- **Target drift** — `target_drift` spec always included.
- **Drift specs** — one set per feature column; routes to parametric (`ttest`) or non-parametric (`mannwhitney`) tests based on normality and sample size; adds `levene` when variance ratio analysis (requires `reference`) indicates heteroscedasticity.
- **Drift specs** — one set per feature column based on column type and sample size.

`reference` is used *only* for variance-ratio analysis inside `suggest()`.  It does not affect the `MonitoringPlan` produced.

Raises `ValueError` for invalid `task_type` or when `reference` is `None`.

### SuggestedPlan

```python
@dataclass
class SuggestedPlan:
    plan: MonitoringPlan          # ready to pass to Runner
    warnings: list[str]           # advisory messages — advisor reasoning per column/metric

    def to_dict(self) -> dict[str, Any]: ...
    # {"plan": plan.model_dump(), "warnings": [...]}
```

### Internal helpers

`analyze_columns(df, schema, reference)` in `_analysis.py` produces one `ColumnAnalysis` per feature column.  `to_float_array(series_native)` in `metrics/tabular/_helpers.py` is shared between `_analysis.py` and `profiler.py` — strips NaNs and returns a `float64` numpy array.

---

## Hiérarchie d'exceptions

```python
class AynError(Exception): ...
class UnknownMetricError(AynError): ...      # métrique non enregistrée
class SchemaError(AynError): ...             # colonne manquante ou type incompatible
class ThresholdError(AynError): ...          # threshold invalide
class InsufficientDataError(AynError): ...   # pas assez de données
class MetricComputeError(AynError): ...      # erreur dans compute() — non-fatale dans Runner
```

---

## Visualisation

- `HtmlRenderer` est une dépendance core (jinja2 + plotly)
- Graphiques via `ChartBackend` Protocol : `PlotlyBackend` (défaut) ou `NoChartBackend` (headless)
- **Pas de SvgBackend** — Plotly est la seule option graphique

```python
HtmlRenderer().render(report)                          # snapshot: histogramme + tableau vert/jaune/rouge
HtmlRenderer().render_history(reports)                 # séries temporelles + bande de seuil + points d'alerte
HtmlRenderer(charts=NoChartBackend()).render(report)   # tableaux seulement
```

**Pas de dashboard built-in** — délégué aux sinks (Grafana, MLflow).

---

## Explicabilité — module explain/

**Niveau 1 — DriftAttributor (v1, zero overhead)**

```python
@dataclass
class DriftContribution:
    feature_name: str
    drift_score: float       # PSI ou Wasserstein de la feature
    importance: float        # feature_importances_ du modèle sklearn
    attribution: float       # drift_score × importance, normalisé [0-1]

@dataclass
class AttributionReport:
    performance_metric: MetricResult
    contributions: list[DriftContribution]
    context: ExecutionContext
```

Usage : `DriftAttributor(report, model_wrapper).attribute("accuracy")` → `AttributionReport`

**Niveau 2 — SHAP distribution monitor (v2, opt-in, `ayn-ml-pro[shap]`)**

Logger les SHAP values sur un échantillon (1–5%) → surveiller leur distribution dans le temps.

**Niveau 3 — LLM-as-judge (v2, opt-in, `ayn-ml[llm-judge]`)**

Via `ragas` ou `deepeval` — pas de réimplémentation.

### Expérimental — explain/experimental/

Marqué `@experimental` — API instable, non couvert par semver.

| Module | Méthode | Priorité |
|--------|---------|----------|
| `distribution_correction.py` | Counterfactuel distribution-level par feature | Premier à implémenter |
| `xpe.py` | XPE — Optimal Transport + Shapley (KDD 2024) | Opportunité unique (pas d'OSS) |
| `token_shap.py` | Attribution token-level LLM | À surveiller (PyPI instable) |
| `agent_causal.py` | Attribution causale agents multi-step | Gap académique non résolu |

---

## Backend et performance

- Preprocessing → narwhals (backend-agnostic)
- Computation → numpy via `.to_numpy()` (scipy/sklearn)
- Zero-copy avec Polars/Arrow — la conversion narwhals → numpy est négligeable

| Contexte | Backend recommandé |
|----------|--------------------|
| Dev / notebooks | pandas |
| Production single-node | Polars |
| Données > RAM | DuckDB + Parquet via DataSource |

**Rust :** bénéfice via Polars uniquement. Pas de code Rust custom en v1.
Voir `docs/research/rust-ml-monitoring-landscape.md`.

---

## Dépendances optionnelles

```toml
[tool.poetry.extras]
polars          = ["polars"]
pandas          = ["pandas"]
sql             = ["sqlalchemy", "pymysql"]
parquet         = ["pyarrow"]
excel           = ["openpyxl"]
mlflow          = ["mlflow"]
nlp             = ["evaluate", "bert-score"]
nlp-embeddings  = ["sentence-transformers"]   # lourd — tire PyTorch
all             = [...]
```

---

## Métriques implémentées

| Catégorie | Métriques | Count |
|-----------|-----------|-------|
| Performance | `accuracy` `precision` `recall` `f1` `log_loss` `auc` `aucpr` `brier` `mse` `mae` `r2` `mape` | 12 |
| Drift | `psi` `wasserstein` `mmd` `target_drift` | 4 |
| Tests statistiques | `ks_2samp` `ttest` `mannwhitney` `levene` `cramervonmises` `chisquare` `hellinger` `jensenshannon` `tvd` `energy_distance` `anderson_darling` `epps_singleton` `fisher_exact` `gtest` `ztest_proportions` | 15 |
| Statistiques | `mean` `median` `std` `skewness` `kurtosis` `quantile` `count` `top_category` `sum` `unique_count` `in_range_count` `out_range_count` `in_list_count` `row_count` `column_count` `almost_constant_columns` `duplicate_rows` `empty_columns` | 18 |
| CBPE | `cbpe_accuracy` `cbpe_auc` `cbpe_f1` `cbpe_precision` `cbpe_recall` | 5 |
| Fairness | `demographic_parity` `equalized_odds` `disparate_impact` | 3 |
| Recsys | `precision_at_k` `recall_at_k` `fbeta_at_k` `hit_rate` `map_at_k` `ndcg_at_k` `mrr_at_k` `diversity` `novelty` `popularity_bias` `personalization` `item_bias` `user_bias` `serendipity` | 14 |
| **Total** | | **71** |

### Notes d'implémentation — à garder en tête

**`mape` :** Mean Absolute *Percentage* Error = `mean(|y_true - y_pred| / |y_true|) * 100`. Différent de MAE.

**PSI :** clipper les valeurs proches de zéro pour éviter division par zéro ou inf/NaN.

**`chisquare` :** attend des fréquences pré-calculées, pas des données brutes. Pré-binning explicite requis.

**`kl_div` (si ajouté) :** `scipy.special.kl_div` retourne un tableau. Wrapper obligatoire avec `sum()` sur distributions normalisées.

**CBPE :** aveugle au concept drift. Toujours coupler avec `psi`/`wasserstein`.

---

## Métriques à implémenter

### Bootstrap de performance — Phase 2c

Bootstrap le reference window pour dériver un intervalle de confiance `[p2.5, p97.5]`, puis vérifie si la valeur du current window tombe en dehors. Résultat dans `MetricResult.conf_interval`.

**Décision ouverte :** logique bootstrap dans `compute()` (métrique auto-contenue) ou dans le Runner (config bootstrap au niveau plan). Trancher via `architect-reviewer`.

### Recsys — Phase 12 (complete — 14/14)

Implemented in `metrics/recsys.py` (flat module): `precision_at_k`, `recall_at_k`, `fbeta_at_k`, `hit_rate`, `map_at_k`, `ndcg_at_k`, `mrr_at_k`, `diversity`, `novelty`, `popularity_bias`, `personalization`, `item_bias`, `user_bias`, `serendipity`.

All operate on the same interactions DataFrame with `RecSysSchema`. `MetricType.recsys` registered.

### NLP — Phase 8 (`pip install ayn-ml[nlp]`)

`bleu`, `rouge1`, `rouge2`, `rougeL`, `bert_score`, `exact_match`, `toxicity`, `embedding_drift`

### Agent — Phase 9

`task_completion_rate`, `tool_success_rate`, `step_count`, `retry_rate`,
`avg_tokens`, `avg_latency_ms`, `avg_cost_usd`, `p95_latency_ms`

### Déféré v2

- DLE (Direct Loss Estimation) — requiert LightGBM, feature_cols, lifecycle stateful
- DiagnosticReport — pattern-matching cross-métrique sur MonitoringReport
- `bftest` (Brown-Forsythe) — initialement prévu dans les tests statistiques, couvert en pratique par `levene` (variante robuste). Déféré v2.

---

## Recherche et références

- `docs/research/rust-ml-monitoring-landscape.md` — pourquoi pas de Rust custom en v1
- `docs/research/ml-monitoring-landscape-2026.md` — analyse concurrentielle, positionnement ayn-ml
- `docs/research/explainability-xai-monitoring-2026.md` — XAI dans le monitoring, XPE, gaps marché
