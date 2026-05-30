"""Metric selection rules for MetricAdvisor.

Internal module.  Not part of the public API.

Each function takes a ``ColumnAnalysis`` (or schema-level context) and
returns ``(specs, warnings)`` — a list of ``MetricSpec`` objects to add to
the plan and a list of human-readable advisory strings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ayn_ml.core.schema import ColumnType
from ayn_ml.core.spec import MetricSpec

if TYPE_CHECKING:
    from ayn_ml.advisor._analysis import ColumnAnalysis
    from ayn_ml.core.schema import TabularSchema

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registered() -> frozenset[str]:
    """Return the set of currently registered metric names.

    Importing here (not at module level) ensures the registry is fully
    populated before the check runs.
    """
    from ayn_ml.metrics import list_metrics  # noqa: PLC0415

    return frozenset(list_metrics())


def _compute_imbalance(df_native: Any, schema: TabularSchema, task_type: str) -> float:
    """Return the class imbalance ratio for a classification task.

    Defined as ``max_class_count / min_class_count``.  Returns ``1.0`` for
    regression tasks, empty label columns, or single-class datasets.

    Args:
        df_native: Narwhals-wrapped current DataFrame.
        schema: TabularSchema for the dataset.
        task_type: ``"classification"`` or ``"regression"``.

    Returns:
        Imbalance ratio ≥ 1.0.
    """
    if task_type != "classification":
        return 1.0
    if schema.label_col not in df_native.columns:
        return 1.0

    arr = df_native[schema.label_col].drop_nulls().to_numpy()
    if len(arr) == 0:
        return 1.0

    _, counts = np.unique(arr, return_counts=True)
    if len(counts) < 2 or counts.min() == 0:
        return 1.0
    return float(counts.max()) / float(counts.min())


# ---------------------------------------------------------------------------
# Drift spec selection per column
# ---------------------------------------------------------------------------


def suggest_drift_specs(
    analysis: ColumnAnalysis,
) -> tuple[list[MetricSpec], list[str]]:
    """Return drift ``MetricSpec`` objects for a single feature column.

    Decision tree (from metric-advisor-selection-guide-2026.md §3.3):

    **Categorical:**  PSI always; chi-square when registered.

    **Numeric/binary — sample-size routing:**

    - n < 30 → wasserstein only (+ warning)
    - n > 50 000 → PSI + wasserstein (no hypothesis tests)
    - otherwise → normal-routing or non-normal routing

    **Normal routing** (is_normal and |skewness| < 1.0):
    ttest (Welch) + Cramér-von Mises + wasserstein + PSI

    **Non-normal routing:**
    Mann-Whitney U + Cramér-von Mises + wasserstein + PSI

    **Variance ratio:** when ``variance_ratio > 1.5`` or ``< 0.67``, Levene
    is added (+ warning).

    Args:
        analysis: Statistical summary for the column.

    Returns:
        Tuple of ``(specs, warnings)``.
    """
    specs: list[MetricSpec] = []
    warnings: list[str] = []
    reg = _registered()
    col = analysis.col_name

    # --- Categorical ---
    if analysis.col_type == ColumnType.categorical:
        specs.append(MetricSpec(name="psi", feature_name=col))
        if "chisquare" in reg:
            specs.append(MetricSpec(name="chisquare", feature_name=col))
        return specs, warnings

    # --- Numeric / binary ---
    n = analysis.n

    if n < 30:
        specs.append(MetricSpec(name="wasserstein", feature_name=col))
        warnings.append(f"'{col}': only wasserstein suggested — n={n} is too small for hypothesis tests (< 30)")
        return specs, warnings

    if n > 50_000:
        _log.debug("'%s': large-n routing (n=%d) — PSI + wasserstein only, no hypothesis tests.", col, n)
        specs.append(MetricSpec(name="psi", feature_name=col))
        specs.append(MetricSpec(name="wasserstein", feature_name=col))
        return specs, warnings

    # Standard numeric routing
    if analysis.is_normal and abs(analysis.skewness) < 1.0:
        _log.debug("'%s': normal routing (skewness=%.2f) — ttest (Welch).", col, analysis.skewness)
        specs.append(MetricSpec(name="ttest", feature_name=col, params={"equal_var": False}))
    else:
        _log.debug(
            "'%s': non-normal routing (skewness=%.2f, is_normal=%s) — mannwhitney.",
            col,
            analysis.skewness,
            analysis.is_normal,
        )
        specs.append(MetricSpec(name="mannwhitney", feature_name=col))

    specs.append(MetricSpec(name="cramervonmises", feature_name=col))
    specs.append(MetricSpec(name="wasserstein", feature_name=col))
    specs.append(MetricSpec(name="psi", feature_name=col))

    # Variance ratio → Levene
    vr = analysis.variance_ratio
    if vr is not None and (vr > 1.5 or vr < 0.67):
        specs.append(MetricSpec(name="levene", feature_name=col))
        warnings.append(f"levene added for '{col}': variance_ratio={vr:.2f}")

    return specs, warnings


# ---------------------------------------------------------------------------
# Performance spec selection
# ---------------------------------------------------------------------------


def suggest_performance_specs(
    schema: TabularSchema,
    task_type: str,
    imbalance_ratio: float,
) -> tuple[list[MetricSpec], list[str]]:
    """Return performance ``MetricSpec`` objects for the monitoring plan.

    **Regression:** ``mae`` + ``r2``.

    **Classification — imbalance routing:**

    - ratio > 10:1 → f1 + aucpr primary; accuracy excluded; warning emitted
    - ratio > 5:1  → f1 + auc primary; accuracy excluded; warning emitted
    - balanced     → accuracy primary

    Args:
        schema: TabularSchema for the dataset.
        task_type: ``"classification"`` or ``"regression"``.
        imbalance_ratio: Pre-computed class imbalance ratio.

    Returns:
        Tuple of ``(specs, warnings)``.
    """
    specs: list[MetricSpec] = []
    warnings: list[str] = []

    if task_type == "regression":
        specs.append(MetricSpec(name="mae"))
        specs.append(MetricSpec(name="r2"))
        return specs, warnings

    # Classification
    if imbalance_ratio > 10:
        warnings.append(f"accuracy excluded: imbalance ratio {imbalance_ratio:.1f}:1 (severe imbalance)")
        specs.append(MetricSpec(name="f1"))
        specs.append(MetricSpec(name="aucpr"))
    elif imbalance_ratio > 5:
        warnings.append(f"accuracy demoted: imbalance ratio {imbalance_ratio:.1f}:1")
        specs.append(MetricSpec(name="f1"))
        specs.append(MetricSpec(name="auc"))
    else:
        # Balanced
        specs.append(MetricSpec(name="accuracy"))

    return specs, warnings
