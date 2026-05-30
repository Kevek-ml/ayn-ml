"""Automatic statistical profiling for tabular DataFrames.

Computes a fixed set of descriptive statistics for each watched column in one
pass.  Intended to be called by the Runner when ``plan.enable_profiling`` is ``True`` —
not part of the metric registry and not configurable via ``MetricSpec``.

Numeric columns (ColumnType.numeric, ColumnType.binary):
    min, max, mean, std, p25, p50, p75, null_count, null_pct

Categorical columns (ColumnType.categorical):
    null_count, null_pct, n_unique, top_category
"""

from __future__ import annotations

import logging
from typing import Any

import narwhals as nw
import numpy as np

from ayn_ml.core.schema import ColumnType, DataSchema
from ayn_ml.metrics.tabular._helpers import classify_columns, to_float_array

_log = logging.getLogger(__name__)

ColumnProfile = dict[str, float | int | str | None]


def profile_columns(df: Any, col_names: list[str], schema: DataSchema) -> dict[str, ColumnProfile]:
    """Compute a statistical profile for each requested column.

    Args:
        df: Current-window DataFrame (any narwhals-compatible eager frame).
        col_names: Column names to profile.  Names absent from ``df`` are
            logged at WARNING level and skipped.
        schema: DataSchema used for column-type classification (respects
            ``TabularSchema.feature_types`` overrides).

    Returns:
        Mapping of column name → profile dict.  Numeric columns carry keys
        ``min``, ``max``, ``mean``, ``std``, ``p25``, ``p50``, ``p75``,
        ``null_count``, ``null_pct``.  Categorical columns carry keys
        ``null_count``, ``null_pct``, ``n_unique``, ``top_category``.
        ``top_category`` is the lexicographically-first category among those
        sharing the highest count (``np.unique`` sorts before ``argmax``).

    Raises:
        TypeError: If ``df`` is a narwhals-incompatible type or a LazyFrame.
    """
    native = nw.from_native(df, eager_only=True)
    col_types = classify_columns(native, schema)
    n_rows = len(native)

    result: dict[str, ColumnProfile] = {}
    for col in col_names:
        if col not in native.columns:
            _log.warning("profile_columns: column %r not found in DataFrame — skipped", col)
            continue

        null_count = int(native[col].is_null().sum())
        null_pct = round(null_count / n_rows, 6) if n_rows else None
        col_type = col_types.get(col, ColumnType.categorical)

        if col_type in (ColumnType.numeric, ColumnType.binary):
            arr = to_float_array(native[col])
            result[col] = {
                "min": float(arr.min()) if len(arr) else None,
                "max": float(arr.max()) if len(arr) else None,
                "mean": float(arr.mean()) if len(arr) else None,
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "p25": float(np.percentile(arr, 25)) if len(arr) else None,
                "p50": float(np.percentile(arr, 50)) if len(arr) else None,
                "p75": float(np.percentile(arr, 75)) if len(arr) else None,
                "null_count": null_count,
                "null_pct": null_pct,
            }
        else:
            arr = native[col].drop_nulls().to_numpy()
            unique_vals, counts = np.unique(arr, return_counts=True) if len(arr) else (np.array([]), np.array([]))
            top = str(unique_vals[np.argmax(counts)]) if len(unique_vals) else None
            result[col] = {
                "null_count": null_count,
                "null_pct": null_pct,
                "n_unique": int(len(unique_vals)),
                "top_category": top,
            }

    return result
