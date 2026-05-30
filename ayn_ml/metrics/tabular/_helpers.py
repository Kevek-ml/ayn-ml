"""Shared helpers for tabular metric modules.

Centralises schema validation, column extraction, and MetricResult
construction used by all tabular metric modules to avoid duplication.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import ColumnType, DataSchema, TabularSchema
from ayn_ml.core.spec import MetricSpec
from ayn_ml.exceptions import InsufficientDataError, SchemaError
from ayn_ml.metrics.base import compute_status


def _result_metric(
    value: float,
    spec: MetricSpec,
    *,
    effect_size: float | None = None,
    effect_size_label: str | None = None,
) -> MetricResult:
    """Build a MetricResult for a scalar float metric value.

    Rounds to 6 decimal places and evaluates pass/fail status against
    ``spec.threshold``.  Used by performance, drift, estimation, statistical
    test, and fairness metrics.

    Args:
        value: Raw metric value (float).
        spec: MetricSpec used to evaluate pass/fail status.
        effect_size: Optional supplementary effect-size measure (e.g. Cohen's
            d, Cliff's delta).  Pass only for statistical test metrics.
        effect_size_label: Identifies the scale of ``effect_size`` (e.g.
            ``"cohen_d"``).  Required when ``effect_size`` is not ``None``.

    Returns:
        MetricResult with value rounded to 6 decimal places.
    """
    rounded = round(value, 6)
    return MetricResult(
        spec=spec,
        value=rounded,
        status=compute_status(rounded, spec),
        effect_size=round(effect_size, 6) if effect_size is not None else None,
        effect_size_label=effect_size_label,
    )


def _result_stat(value: float | int | str, spec: MetricSpec) -> MetricResult:
    """Build a MetricResult for a descriptive statistic value.

    Accepts ``float``, ``int``, or ``str`` values.  String values (e.g. from
    ``top_category``) always yield ``status=None`` because threshold
    comparison is undefined for non-numeric outputs.  Numeric values are
    evaluated against ``spec.threshold`` via ``compute_status``.

    Does **not** round the value — ``int`` and ``str`` must be preserved
    as-is; rounding a string raises ``TypeError``.

    Args:
        value: Computed statistic (float, int, or str).
        spec: MetricSpec used for pass/fail status evaluation.

    Returns:
        MetricResult with the raw value and appropriate status.
    """
    status = compute_status(value, spec) if not isinstance(value, str) else None
    return MetricResult(spec=spec, value=value, status=status)


def _check_tabular(schema: DataSchema) -> TabularSchema:
    """Assert that the schema is a TabularSchema and return it.

    Args:
        schema: Any DataSchema variant.

    Returns:
        The same object cast to TabularSchema.

    Raises:
        SchemaError: If ``schema`` is not a TabularSchema instance.
    """
    if not isinstance(schema, TabularSchema):
        raise SchemaError(f"Expected TabularSchema, got {type(schema).__name__}.")
    return schema


def _extract_proba(df: Any, schema: TabularSchema, window: str) -> np.ndarray:
    """Extract ``y_pred_proba`` from a DataFrame as a float array.

    Args:
        df: Input DataFrame (any narwhals-compatible frame).
        schema: TabularSchema with ``proba_col`` set.
        window: Human-readable label used in error messages
            (e.g. ``"current"`` or ``"reference"``).

    Returns:
        1-D float array of predicted probabilities.

    Raises:
        SchemaError: If ``proba_col`` is ``None`` or absent from the frame.
    """
    native = nw.from_native(df, eager_only=True)
    if not schema.proba_col or schema.proba_col not in native.columns:
        raise SchemaError(f"proba_col '{schema.proba_col}' not found in the {window} window.")
    return native[schema.proba_col].to_numpy().astype(float)


def _extract_label(df: Any, schema: TabularSchema, window: str) -> np.ndarray:
    """Extract the label column from a DataFrame.

    Args:
        df: Input DataFrame (any narwhals-compatible frame).
        schema: TabularSchema with ``label_col`` set.
        window: Human-readable label for error messages
            (e.g. ``"reference"`` or ``"current"``).

    Returns:
        1-D numpy array of label values preserving the column's native dtype.

    Raises:
        SchemaError: If ``label_col`` is absent from the frame.
    """
    native = nw.from_native(df, eager_only=True)
    if schema.label_col not in native.columns:
        raise SchemaError(f"label_col '{schema.label_col}' not found in the {window} window.")
    return native[schema.label_col].to_numpy()


def _extract_pred(df: Any, schema: TabularSchema, window: str = "current") -> np.ndarray:
    """Extract ``y_pred`` from a DataFrame.

    Args:
        df: Input DataFrame (any narwhals-compatible frame).
        schema: TabularSchema with ``prediction_col`` set.
        window: Human-readable label used in error messages
            (e.g. ``"current"`` or ``"reference"``).

    Returns:
        1-D array of predictions.

    Raises:
        SchemaError: If ``prediction_col`` is absent from the frame.
    """
    native = nw.from_native(df, eager_only=True)
    if schema.prediction_col not in native.columns:
        raise SchemaError(f"Column '{schema.prediction_col}' not found in the {window} window.")
    return native[schema.prediction_col].to_numpy()


def extract_feature(
    df: Any,
    feature_name: str | None,
    *,
    as_float: bool = True,
    min_rows: int = 0,
) -> np.ndarray:
    """Extract a feature column from a DataFrame as a numpy array.

    Canonical extraction helper shared by statistics, statistical-test, drift,
    and fairness modules.  Centralises the four-step guard: not-None check,
    column-presence check, dtype cast, and minimum-row check.

    Args:
        df: Input DataFrame (any narwhals-compatible frame).
        feature_name: Column to extract.  Must not be ``None``.
        as_float: When ``True`` (default), cast the array to ``float64``.
            Pass ``False`` to preserve the column's native dtype (e.g. for
            categorical columns where dtype carries meaning).
        min_rows: Minimum number of rows required.  ``0`` disables the check.
            Raises ``InsufficientDataError`` when the array has fewer rows.

    Returns:
        1-D numpy array; dtype is ``float64`` when ``as_float=True``,
        otherwise the column's native dtype.

    Raises:
        SchemaError: If ``feature_name`` is ``None`` or the column is absent.
        InsufficientDataError: If the array has fewer than ``min_rows`` rows.
    """
    if feature_name is None:
        raise SchemaError("feature_name must not be None.")
    native = nw.from_native(df, eager_only=True)
    if feature_name not in native.columns:
        raise SchemaError(f"Column '{feature_name}' not found.")
    arr = native[feature_name].to_numpy()
    if as_float:
        arr = arr.astype(float)
    if min_rows > 0 and len(arr) < min_rows:
        raise InsufficientDataError(f"At least {min_rows} rows required.")
    return arr


def _require_reference(reference: Any, name: str) -> None:
    """Raise ``SchemaError`` if reference data is ``None``.

    Shared guard used by metrics that require a reference window.  Replaces
    the repeated ``if reference is None: raise SchemaError(...)`` pattern
    spread across drift and statistical-test modules.

    Args:
        reference: Reference window (any value; only checked for ``None``).
        name: Metric name used in the error message.

    Raises:
        SchemaError: If ``reference`` is ``None``.
    """
    if reference is None:
        raise SchemaError(f"{name} requires reference data.")


def _is_numeric(arr: np.ndarray, feature_name: str, schema: DataSchema) -> bool:
    """Return True if a feature should be treated as numeric.

    Resolution order:

    1. ``schema.feature_types[feature_name]`` — explicit declaration wins.
    2. ``arr.dtype`` — falls back to numpy dtype inference when no declaration
       exists.  Correct for float/string columns; may be wrong for
       integer-encoded categoricals, which is why explicit declaration exists.

    Args:
        arr: Extracted numpy array for the feature.
        feature_name: Name of the feature column.
        schema: DataSchema; ``TabularSchema.feature_types`` is consulted when
            present.

    Returns:
        ``True`` if the feature should be treated as numeric, ``False`` if
        categorical.
    """
    declared = getattr(schema, "feature_types", {}).get(feature_name)
    if declared == "categorical":
        return False
    if declared == "numeric":
        return True
    return bool(np.issubdtype(arr.dtype, np.number))


def classify_columns(df: Any, schema: DataSchema) -> dict[str, ColumnType]:
    """Classify every column in ``df`` by its ``ColumnType``.

    Intended to be called **once per window** before the metric loop so that
    N metrics running on the same column do not each re-extract and re-classify
    it.  The result is a flat mapping that the Runner consults for compatibility
    routing.

    Resolution order (per column):

    1. ``schema.feature_types`` explicit declaration → ``numeric`` or
       ``categorical`` directly.
    2. numpy dtype inference:
       - Integer dtype with unique values ``⊆ {0, 1}`` → ``binary``.
       - Any other numeric dtype (float or multi-valued int) → ``numeric``.
       - All other dtypes (object, string) → ``categorical``.

    Args:
        df: Input DataFrame (any narwhals-compatible eager frame).
        schema: DataSchema; ``TabularSchema.feature_types`` overrides inference.

    Returns:
        Mapping of column name → ``ColumnType`` for every column present in
        ``df``.  Classification is derived from the current window only;
        reference-window type discrepancies are not checked at this stage.

    Raises:
        TypeError: If ``df`` is a narwhals-incompatible type or a LazyFrame.
    """
    native = nw.from_native(df, eager_only=True)
    feature_types = getattr(schema, "feature_types", {})
    result: dict[str, ColumnType] = {}
    for col_name in native.columns:
        declared = feature_types.get(col_name)
        if declared == "categorical":
            result[col_name] = ColumnType.categorical
            continue
        if declared == "numeric":
            result[col_name] = ColumnType.numeric
            continue
        arr = native[col_name].to_numpy()
        if np.issubdtype(arr.dtype, np.integer):
            unique = np.unique(arr)
            result[col_name] = ColumnType.binary if set(unique).issubset({0, 1}) else ColumnType.numeric
        elif np.issubdtype(arr.dtype, np.floating):
            # Nullable integer columns (pandas Int64) surface as float when NaN is present.
            unique_non_nan = set(arr[~np.isnan(arr)].tolist())
            # Empty set (all-NaN column) is a subset of everything — guard explicitly.
            result[col_name] = (
                ColumnType.binary if unique_non_nan and unique_non_nan.issubset({0.0, 1.0}) else ColumnType.numeric
            )
        elif np.issubdtype(arr.dtype, np.number):
            result[col_name] = ColumnType.numeric
        else:
            result[col_name] = ColumnType.categorical
    return result


def to_float_array(series_native: Any) -> np.ndarray:
    """Extract a narwhals Series as a float64 numpy array with NaNs removed.

    Shared by the profiler and the MetricAdvisor advisor so both use the
    same null-stripping convention.

    Args:
        series_native: A narwhals Series (already extracted from a frame).

    Returns:
        1-D float64 array with no NaN values.
    """
    arr = np.asarray(series_native.to_numpy(), dtype=float)
    return arr[~np.isnan(arr)]
