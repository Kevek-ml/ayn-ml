"""Column analysis for MetricAdvisor — normality tests and variance ratio.

Internal module.  Not part of the public API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np

from ayn_ml.core.schema import ColumnType
from ayn_ml.metrics.tabular._helpers import classify_columns, to_float_array

if TYPE_CHECKING:
    from ayn_ml.core.schema import TabularSchema

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnAnalysis:
    """Statistical summary of a single feature column.

    Used by ``_rules.py`` to route each feature to the appropriate drift
    metrics.  Numeric stats (skewness, kurtosis, is_normal) are set to
    neutral defaults for categorical columns.

    Args:
        col_name: Column name as it appears in the DataFrame.
        col_type: Inferred or declared column type.
        n: Number of non-null rows used for analysis.
        skewness: Sample skewness (0.0 for categoricals).
        kurtosis: Excess kurtosis (0.0 for categoricals).
        is_normal: Whether the normality test passed at α=0.05.
            Always ``False`` for categoricals.
        variance_ratio: ``std_current / std_reference``.  ``None`` when no
            reference was provided or when std_reference is zero.
    """

    col_name: str
    col_type: ColumnType
    n: int
    skewness: float
    kurtosis: float
    is_normal: bool
    variance_ratio: float | None


def _is_normal(arr: np.ndarray, skewness: float) -> bool:
    """Run the appropriate normality test for the sample size.

    Thresholds (from metric-advisor-selection-guide-2026.md §3.3):

    - n ≤ 300  → Shapiro-Wilk (most powerful for small samples; unreliable
      above n ≈ 1 000, so the cutoff is intentionally conservative)
    - n ≤ 5 000 → D'Agostino k² (``scipy.stats.normaltest``)
    - n > 5 000 → ``|skewness| < 1.0`` heuristic (tests at this scale flag
      almost everything due to statistical power)

    Args:
        arr: Non-NaN float array.
        skewness: Pre-computed sample skewness.

    Returns:
        ``True`` when the normality hypothesis is not rejected (p > 0.05
        for statistical tests, or |skewness| < 1.0 for the heuristic).
    """
    from scipy import stats  # noqa: PLC0415

    n = len(arr)
    if n < 8:
        # Too few points for any test — assume non-normal
        return False
    if n <= 300:
        _, p = stats.shapiro(arr)
        return bool(p > 0.05)
    if n <= 5_000:
        _, p = stats.normaltest(arr)
        return bool(p > 0.05)
    return bool(abs(skewness) < 1.0)


def _variance_ratio(cur_arr: np.ndarray, ref_native: Any, col: str) -> float | None:
    """Compute std_current / std_reference for a numeric column.

    Both stds use ddof=1 (sample std), matching the profiler convention.
    The ratio is mathematically invariant to the ddof choice when both sides
    use the same value, but ddof=1 is used throughout for consistency.

    Args:
        cur_arr: Non-NaN current-window float array.
        ref_native: Narwhals-wrapped reference DataFrame.
        col: Column name.

    Returns:
        Ratio as a float, or ``None`` when reference std is zero or the
        column is absent from the reference.
    """
    if col not in ref_native.columns:
        return None
    ref_arr = to_float_array(ref_native[col])
    if len(ref_arr) == 0:
        return None
    ref_std = float(np.std(ref_arr, ddof=1)) if len(ref_arr) > 1 else 0.0
    if ref_std == 0.0:
        return None
    cur_std = float(np.std(cur_arr, ddof=1)) if len(cur_arr) > 1 else 0.0
    return cur_std / ref_std


def _feature_cols(df_native: Any, schema: TabularSchema) -> list[str]:
    """Return the list of feature columns to analyse.

    Excludes label, prediction, probability, timestamp, model_id, and
    model_version columns.  When ``schema.feature_types`` is non-empty it
    is used as the authoritative list; otherwise all non-special columns
    from the DataFrame are used.

    Args:
        df_native: Narwhals-wrapped current DataFrame.
        schema: TabularSchema describing the data.

    Returns:
        Ordered list of feature column names.
    """
    special: set[str] = {schema.label_col, schema.prediction_col}
    if schema.proba_col:
        special.add(schema.proba_col)
    if schema.timestamp_col:
        special.add(schema.timestamp_col)
    if schema.model_id_col:
        special.add(schema.model_id_col)
    if schema.model_version_col:
        special.add(schema.model_version_col)

    if schema.feature_types:
        return [c for c in schema.feature_types if c not in special]
    return [c for c in df_native.columns if c not in special]


def analyze_columns(
    df: Any,
    schema: TabularSchema,
    reference: Any | None = None,
) -> list[ColumnAnalysis]:
    """Analyse each feature column and return statistical summaries.

    For numeric/binary columns the function computes skewness, kurtosis,
    runs a normality test, and optionally the variance ratio against
    ``reference``.  Normality, skewness, and kurtosis are computed on the
    **reference** distribution (the stable baseline) when the column is
    present in ``reference``; they fall back to the current window when the
    column is absent from ``reference`` or ``reference`` is ``None``.
    Sample-size routing always uses the current window.
    For categorical columns only ``n`` is recorded.

    Args:
        df: Current-window DataFrame (narwhals-compatible).
        schema: TabularSchema for the dataset.
        reference: Optional reference DataFrame.  When provided, enables
            ``variance_ratio`` computation and hence Levene routing.

    Returns:
        One ``ColumnAnalysis`` per feature column, in column order.
    """
    nw_df = nw.from_native(df, eager_only=True)
    nw_ref = nw.from_native(reference, eager_only=True) if reference is not None else None

    col_types = classify_columns(nw_df, schema)
    feature_cols = _feature_cols(nw_df, schema)
    n_rows = len(nw_df)

    analyses: list[ColumnAnalysis] = []
    for col in feature_cols:
        if col not in nw_df.columns:
            _log.debug("MetricAdvisor: column '%s' declared in schema but absent from df; skipping.", col)
            continue

        col_type = col_types.get(col, ColumnType.categorical)

        if col_type == ColumnType.categorical:
            analyses.append(
                ColumnAnalysis(
                    col_name=col,
                    col_type=col_type,
                    n=n_rows,
                    skewness=0.0,
                    kurtosis=0.0,
                    is_normal=False,
                    variance_ratio=None,
                )
            )
            continue

        # Numeric / binary
        arr = to_float_array(nw_df[col])
        n = len(arr)  # sample size routing always uses the current window

        # Normality routing uses the reference distribution — the stable baseline we
        # are monitoring against.  Using the potentially-drifted current window would
        # be circular (drift could make it non-normal, silently changing the test
        # choice).  Fall back to current only when the column is absent from the
        # reference or the reference array is empty.
        if nw_ref is not None and col in nw_ref.columns:
            norm_arr = to_float_array(nw_ref[col])
            if len(norm_arr) == 0:
                _log.debug("'%s': reference array empty — normality routing falls back to current window.", col)
                norm_arr = arr
        else:
            norm_arr = arr

        if len(norm_arr) > 0:
            from scipy import stats  # noqa: PLC0415

            skewness = float(stats.skew(norm_arr))
            kurt = float(stats.kurtosis(norm_arr))
            normal = _is_normal(norm_arr, skewness)
        else:
            skewness = 0.0
            kurt = 0.0
            normal = False

        vr = _variance_ratio(arr, nw_ref, col) if nw_ref is not None else None

        analyses.append(
            ColumnAnalysis(
                col_name=col,
                col_type=col_type,
                n=n,
                skewness=skewness,
                kurtosis=kurt,
                is_normal=normal,
                variance_ratio=vr,
            )
        )

    return analyses
