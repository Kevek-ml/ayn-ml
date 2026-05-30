"""Fairness and bias metrics for tabular ML models.

Covers group-level disparity in predictions: demographic parity difference,
equalized odds difference, and disparate impact ratio.  All three metrics
operate on the current window only (no reference required) and identify the
protected attribute via ``spec.feature_name``, which must point to a column
declared in ``TabularSchema.protected_cols`` when that field is set.

Metric naming convention mirrors Fairlearn's terminology to ease migration.
"""

from __future__ import annotations

import logging
from typing import Any

import narwhals as nw
import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import ColumnType, DataSchema, TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import InsufficientDataError, SchemaError
from ayn_ml.metrics.registry import register_metric
from ayn_ml.metrics.tabular._helpers import _check_tabular, _result_metric

_logger = logging.getLogger(__name__)

_MIN_ROWS = 10


def _extract_protected(df: nw.DataFrame, spec: MetricSpec, schema: TabularSchema) -> np.ndarray:
    """Extract and validate the protected attribute column.

    Args:
        df: Narwhals DataFrame.
        spec: MetricSpec whose ``feature_name`` names the protected column.
        schema: TabularSchema with optional ``protected_cols`` declaration.

    Returns:
        1-D numpy array of group labels.

    Raises:
        SchemaError: If ``feature_name`` is None, not declared in
            ``protected_cols``, or absent from the DataFrame.
    """
    col = spec.feature_name
    if col is None:
        raise SchemaError(
            "feature_name is required for fairness metrics — set it to the protected attribute column name."
        )
    if schema.protected_cols is not None and col not in schema.protected_cols:
        raise SchemaError(
            f"Column '{col}' is not declared in TabularSchema.protected_cols. "
            f"Declared columns: {schema.protected_cols}."
        )
    if col not in df.columns:
        raise SchemaError(f"Protected column '{col}' not found in DataFrame.")
    return df[col].to_numpy()


@register_metric("demographic_parity")
class DemographicParityMetric:
    r"""Demographic parity difference across protected groups.

    Measures the maximum difference in positive prediction rates between any
    two groups defined by the protected attribute.  A value of 0 means all
    groups receive positive predictions at the same rate (perfect parity).
    Lower is better — set ``spec.upper_bound=True`` with a threshold such as
    0.1 or 0.2 to flag unacceptable disparities.

    .. math::

        \\text{DPD} = \\max_g P(\\hat{Y}=1 \\mid A=g)
                     - \\min_g P(\\hat{Y}=1 \\mid A=g)

    Uses ``spec.feature_name`` to identify the protected attribute column.
    The column must be declared in ``TabularSchema.protected_cols`` when that
    field is set.  Only the current window is used; no reference required.

    Raises:
        SchemaError: If ``feature_name`` is None, undeclared, or absent.
        InsufficientDataError: If the DataFrame has fewer than 10 rows.
    """

    name = "demographic_parity"
    metric_type = MetricType.fairness
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.categorical, ColumnType.binary})
    accepted_target_types: dict[str, frozenset[ColumnType]] = {
        "prediction_col": frozenset({ColumnType.binary}),
    }

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Compute demographic parity difference on the current window.

        Args:
            current: Current-window DataFrame (any narwhals-compatible frame).
            reference: Unused; accepted for protocol compatibility.
            schema: Data schema describing column roles.
            spec: MetricSpec with ``feature_name`` pointing to the protected
                attribute column.

        Returns:
            MetricResult where ``value`` is the demographic parity difference.
        """
        tab = _check_tabular(schema)
        df = nw.from_native(current, eager_only=True)

        if len(df) < _MIN_ROWS:
            raise InsufficientDataError(f"At least {_MIN_ROWS} rows required for fairness metrics.")
        if tab.prediction_col not in df.columns:
            raise SchemaError(f"Column '{tab.prediction_col}' not found in DataFrame.")

        groups = _extract_protected(df, spec, tab)
        y_pred = df[tab.prediction_col].to_numpy()

        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            _logger.warning(
                "demographic_parity: only one group found in '%s'; returning 0.",
                spec.feature_name,
            )
            return _result_metric(0.0, spec)

        rates = np.array([y_pred[groups == g].mean() for g in unique_groups], dtype=float)
        return _result_metric(float(rates.max() - rates.min()), spec)


@register_metric("equalized_odds")
class EqualizedOddsMetric:
    r"""Equalized odds difference across protected groups.

    Measures the maximum gap in true positive rates (TPR) and false positive
    rates (FPR) between groups, taking the larger of the two gaps.  A value
    of 0 means all groups have equal TPR *and* FPR (perfect equalized odds).
    Lower is better — set ``spec.upper_bound=True`` with a threshold such as
    0.1.

    .. math::

        \\text{EOD} = \\max\\bigl(
            \\max_g \\text{TPR}_g - \\min_g \\text{TPR}_g,\\;
            \\max_g \\text{FPR}_g - \\min_g \\text{FPR}_g
        \\bigr)

    Uses ``spec.feature_name`` to identify the protected attribute column.
    Requires both ``y_true`` (``schema.label_col``) and ``y_pred``
    (``schema.prediction_col``).  Groups with no positive (or no negative)
    labels contribute ``nan`` to the TPR (or FPR) array and are excluded from
    the gap via ``np.nanmax`` / ``np.nanmin``.

    Raises:
        SchemaError: If ``feature_name`` is None, undeclared, or absent, or if
            required label/prediction columns are missing.
        InsufficientDataError: If the DataFrame has fewer than 10 rows.
    """

    name = "equalized_odds"
    metric_type = MetricType.fairness
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.categorical, ColumnType.binary})
    accepted_target_types: dict[str, frozenset[ColumnType]] = {
        "prediction_col": frozenset({ColumnType.binary}),
        "label_col": frozenset({ColumnType.binary}),
    }

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Compute equalized odds difference on the current window.

        Args:
            current: Current-window DataFrame (any narwhals-compatible frame).
            reference: Unused; accepted for protocol compatibility.
            schema: Data schema describing column roles.
            spec: MetricSpec with ``feature_name`` pointing to the protected
                attribute column.

        Returns:
            MetricResult where ``value`` is the equalized odds difference.
        """
        tab = _check_tabular(schema)
        df = nw.from_native(current, eager_only=True)

        if len(df) < _MIN_ROWS:
            raise InsufficientDataError(f"At least {_MIN_ROWS} rows required for fairness metrics.")
        for col in (tab.label_col, tab.prediction_col):
            if col not in df.columns:
                raise SchemaError(f"Column '{col}' not found in DataFrame.")

        groups = _extract_protected(df, spec, tab)
        y_true = df[tab.label_col].to_numpy().astype(float)
        y_pred = df[tab.prediction_col].to_numpy()

        unique_labels = np.unique(y_true)
        if not set(unique_labels.tolist()).issubset({0.0, 1.0}):
            raise SchemaError(
                f"equalized_odds requires binary labels (0/1); "
                f"found: {unique_labels.tolist()}. "
                "For multi-class, compute per-class TPR/FPR separately."
            )

        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            _logger.warning(
                "equalized_odds: only one group found in '%s'; returning 0.",
                spec.feature_name,
            )
            return _result_metric(0.0, spec)

        tprs, fprs = [], []
        for g in unique_groups:
            mask = groups == g
            yt, yp = y_true[mask], y_pred[mask]
            pos_mask = yt == 1.0
            neg_mask = yt == 0.0
            tprs.append(float(yp[pos_mask].mean()) if pos_mask.any() else float("nan"))
            fprs.append(float(yp[neg_mask].mean()) if neg_mask.any() else float("nan"))

        tprs_arr = np.array(tprs)
        fprs_arr = np.array(fprs)

        tpr_all_nan = bool(np.all(np.isnan(tprs_arr)))
        fpr_all_nan = bool(np.all(np.isnan(fprs_arr)))
        if tpr_all_nan and fpr_all_nan:
            _logger.warning(
                "equalized_odds: no group has both positive and negative labels; "
                "TPR and FPR are both undefined — returning 0."
            )
            return _result_metric(0.0, spec)

        tpr_gap = float(np.nanmax(tprs_arr) - np.nanmin(tprs_arr)) if not tpr_all_nan else 0.0
        fpr_gap = float(np.nanmax(fprs_arr) - np.nanmin(fprs_arr)) if not fpr_all_nan else 0.0

        return _result_metric(max(tpr_gap, fpr_gap), spec)


@register_metric("disparate_impact")
class DisparateImpactMetric:
    r"""Disparate impact ratio across protected groups.

    Ratio of the lowest to the highest positive prediction rate among groups.
    A value of 1.0 means identical rates across groups (no disparity).  The
    80% rule (US EEOC guideline) flags values below 0.8 as potentially
    discriminatory.  Higher is better — set ``spec.upper_bound=False`` with
    ``spec.threshold=0.8`` to apply the 80% rule.

    .. math::

        \\text{DIR} = \\frac{\\min_g P(\\hat{Y}=1 \\mid A=g)}
                           {\\max_g P(\\hat{Y}=1 \\mid A=g)}

    Uses ``spec.feature_name`` to identify the protected attribute column.
    Returns 1.0 (no disparity) when all groups have a near-zero positive
    prediction rate (denominator ≈ 0) or when only one group is present.

    Raises:
        SchemaError: If ``feature_name`` is None, undeclared, or absent.
        InsufficientDataError: If the DataFrame has fewer than 10 rows.
    """

    name = "disparate_impact"
    metric_type = MetricType.fairness
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.categorical, ColumnType.binary})
    accepted_target_types: dict[str, frozenset[ColumnType]] = {
        "prediction_col": frozenset({ColumnType.binary}),
    }

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Compute disparate impact ratio on the current window.

        Args:
            current: Current-window DataFrame (any narwhals-compatible frame).
            reference: Unused; accepted for protocol compatibility.
            schema: Data schema describing column roles.
            spec: MetricSpec with ``feature_name`` pointing to the protected
                attribute column.  Set ``upper_bound=False`` and
                ``threshold=0.8`` to apply the 80% rule.

        Returns:
            MetricResult where ``value`` is the disparate impact ratio in
            [0, 1].
        """
        tab = _check_tabular(schema)
        df = nw.from_native(current, eager_only=True)

        if len(df) < _MIN_ROWS:
            raise InsufficientDataError(f"At least {_MIN_ROWS} rows required for fairness metrics.")
        if tab.prediction_col not in df.columns:
            raise SchemaError(f"Column '{tab.prediction_col}' not found in DataFrame.")

        groups = _extract_protected(df, spec, tab)
        y_pred = df[tab.prediction_col].to_numpy()

        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            _logger.warning(
                "disparate_impact: only one group found in '%s'; returning 1.0.",
                spec.feature_name,
            )
            return _result_metric(1.0, spec)

        rates = np.array([y_pred[groups == g].mean() for g in unique_groups], dtype=float)
        max_rate = rates.max()
        if max_rate < 1e-9:
            _logger.warning("disparate_impact: all groups have near-zero positive prediction rate; returning 1.0.")
            return _result_metric(1.0, spec)

        return _result_metric(float(rates.min() / max_rate), spec)
