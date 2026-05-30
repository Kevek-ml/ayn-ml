"""Supervised performance metrics for tabular ML models.

Covers classification (accuracy, precision, recall, F1, log-loss, AUC,
AUCPR, Brier) and regression (MSE, MAE, R², MAPE).  All metrics are
stateless, require no reference window, and delegate to scikit-learn which
is imported lazily inside each ``compute()`` to keep cold-import time low.
"""

from __future__ import annotations

import logging
from typing import Any

import narwhals as nw
import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import DataSchema, TabularSchema
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import InsufficientDataError, MetricComputeError, SchemaError
from ayn_ml.metrics.registry import register_metric
from ayn_ml.metrics.tabular._helpers import _check_tabular, _result_metric

_logger = logging.getLogger(__name__)


def _extract_arrays(
    current: Any,
    schema: TabularSchema,
    need_proba: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract y_true, y_pred, and optionally y_proba from a DataFrame.

    Args:
        current: Current-window DataFrame (any narwhals-compatible frame).
        schema: TabularSchema describing label and prediction column names.
        need_proba: When ``True``, also extracts the probability column
            and raises if it is absent.

    Returns:
        A three-tuple ``(y_true, y_pred, y_proba)`` where ``y_proba`` is
        ``None`` when ``need_proba=False``.

    Raises:
        SchemaError: If ``label_col`` or ``prediction_col`` is missing, or if
            ``need_proba=True`` and ``proba_col`` is absent or ``None``.
        InsufficientDataError: If the DataFrame has fewer than 2 rows.
    """
    df = nw.from_native(current, eager_only=True)

    for col in (schema.label_col, schema.prediction_col):
        if col not in df.columns:
            raise SchemaError(f"Column '{col}' not found in DataFrame.")

    if len(df) < 2:
        raise InsufficientDataError("At least 2 rows required to compute a metric.")

    y_true = df[schema.label_col].to_numpy()
    y_pred = df[schema.prediction_col].to_numpy()
    y_proba: np.ndarray | None = None

    if need_proba:
        if schema.proba_col and schema.proba_col in df.columns:
            y_proba = df[schema.proba_col].to_numpy()
        else:
            raise SchemaError(f"Probability column '{schema.proba_col}' required but not found.")

    return y_true, y_pred, y_proba


@register_metric("accuracy")
class AccuracyMetric:
    """Fraction of correctly classified samples.

    Computed as ``(y_true == y_pred).mean()``.  Works for both binary and
    multi-class classification.
    """

    name = "accuracy"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute accuracy.

        Args:
            current: Current-window DataFrame.
            reference: Ignored (not required).
            schema: TabularSchema with label and prediction column names.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value in [0, 1].
        """
        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        return _result_metric(float((y_true == y_pred).mean()), spec)


@register_metric("precision")
class PrecisionMetric:
    """Precision (positive predictive value) for classification.

    Delegates to ``sklearn.metrics.precision_score``.  Averaging strategy
    can be overridden via ``spec.params["average"]`` (default ``"weighted"``).
    """

    name = "precision"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute precision.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema.
            spec: MetricSpec; ``params["average"]`` controls averaging
                (``"weighted"``, ``"macro"``, ``"binary"``, etc.).

        Returns:
            MetricResult with value in [0, 1].
        """
        from sklearn.metrics import precision_score

        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        average = spec.params.get("average", "weighted")
        value = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        return _result_metric(value, spec)


@register_metric("recall")
class RecallMetric:
    """Recall (sensitivity / true-positive rate) for classification.

    Delegates to ``sklearn.metrics.recall_score``.
    """

    name = "recall"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute recall.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema.
            spec: MetricSpec; ``params["average"]`` controls averaging
                (default ``"weighted"``).

        Returns:
            MetricResult with value in [0, 1].
        """
        from sklearn.metrics import recall_score

        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        average = spec.params.get("average", "weighted")
        value = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        return _result_metric(value, spec)


@register_metric("f1")
class F1Metric:
    """F1 score (harmonic mean of precision and recall).

    Delegates to ``sklearn.metrics.f1_score``.
    """

    name = "f1"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute F1 score.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema.
            spec: MetricSpec; ``params["average"]`` controls averaging
                (default ``"weighted"``).

        Returns:
            MetricResult with value in [0, 1].
        """
        from sklearn.metrics import f1_score

        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        average = spec.params.get("average", "weighted")
        value = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        return _result_metric(value, spec)


@register_metric("log_loss")
class LogLossMetric:
    """Logarithmic loss (cross-entropy) for probabilistic classifiers.

    Requires ``TabularSchema.proba_col`` to be set.
    """

    name = "log_loss"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:  # noqa: DOC502
        """Compute log-loss.

        Args:
            current: Current-window DataFrame including a probability column.
            reference: Ignored.
            schema: TabularSchema with ``proba_col`` set.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value > 0.

        Raises:
            SchemaError: If the probability column is missing.
        """
        from sklearn.metrics import log_loss

        s = _check_tabular(schema)
        y_true, _, y_proba = _extract_arrays(current, s, need_proba=True)
        value = float(log_loss(y_true, y_proba))
        return _result_metric(value, spec)


@register_metric("auc")
class AUCMetric:
    """Area Under the ROC Curve (AUROC).

    Requires ``TabularSchema.proba_col``.  For multi-class problems set
    ``spec.params["multi_class"]`` to ``"ovr"`` or ``"ovo"``.
    """

    name = "auc"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:  # noqa: DOC502
        """Compute AUROC.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema with ``proba_col`` set.
            spec: MetricSpec; ``params["multi_class"]`` overrides the
                ``roc_auc_score`` ``multi_class`` argument (default ``"raise"``).

        Returns:
            MetricResult with value in [0, 1].

        Raises:
            SchemaError: If the probability column is missing.
        """
        from sklearn.metrics import roc_auc_score

        s = _check_tabular(schema)
        y_true, _, y_proba = _extract_arrays(current, s, need_proba=True)
        if len(np.unique(y_true)) < 2:
            raise MetricComputeError("auc: ROC AUC is undefined when only one class is present.")
        multi_class = spec.params.get("multi_class", "raise")
        try:
            value = float(roc_auc_score(y_true, y_proba, multi_class=multi_class))
        except ValueError as exc:
            raise MetricComputeError(f"auc: {exc}") from exc
        return _result_metric(value, spec)


@register_metric("aucpr")
class AUCPRMetric:
    """Area Under the Precision-Recall Curve (Average Precision).

    Requires ``TabularSchema.proba_col``.  More informative than AUROC on
    imbalanced datasets.
    """

    name = "aucpr"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:  # noqa: DOC502
        """Compute average precision (AUCPR).

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema with ``proba_col`` set.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value in [0, 1].

        Raises:
            SchemaError: If the probability column is missing.
        """
        from sklearn.metrics import average_precision_score

        s = _check_tabular(schema)
        y_true, _, y_proba = _extract_arrays(current, s, need_proba=True)
        if len(np.unique(y_true)) < 2:
            raise MetricComputeError("aucpr: precision-recall AUC is undefined when only one class is present.")
        try:
            value = float(average_precision_score(y_true, y_proba))
        except ValueError as exc:
            raise MetricComputeError(f"aucpr: {exc}") from exc
        return _result_metric(value, spec)


@register_metric("brier")
class BrierMetric:
    """Brier score: mean squared error between probabilities and outcomes.

    Lower is better; 0 is perfect, 1 is worst.  Requires ``proba_col``.
    """

    name = "brier"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:  # noqa: DOC502
        """Compute Brier score.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema with ``proba_col`` set.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value in [0, 1].

        Raises:
            SchemaError: If the probability column is missing.
        """
        from sklearn.metrics import brier_score_loss

        s = _check_tabular(schema)
        y_true, _, y_proba = _extract_arrays(current, s, need_proba=True)
        value = float(brier_score_loss(y_true, y_proba))
        return _result_metric(value, spec)


@register_metric("mse")
class MSEMetric:
    """Mean Squared Error for regression models."""

    name = "mse"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute MSE.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value >= 0.
        """
        from sklearn.metrics import mean_squared_error

        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        value = float(mean_squared_error(y_true, y_pred))
        return _result_metric(value, spec)


@register_metric("mae")
class MAEMetric:
    """Mean Absolute Error for regression models."""

    name = "mae"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute MAE.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value >= 0.
        """
        from sklearn.metrics import mean_absolute_error

        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        value = float(mean_absolute_error(y_true, y_pred))
        return _result_metric(value, spec)


@register_metric("r2")
class R2Metric:
    """Coefficient of determination (R²) for regression models.

    Ranges from −∞ to 1; 1 is perfect fit, 0 means the model is no better
    than predicting the mean.
    """

    name = "r2"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute R².

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value ≤ 1.

        Raises:
            MetricComputeError: If all ``y_true`` values are identical
                (zero variance makes R² undefined).
        """
        from sklearn.metrics import r2_score

        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        if np.unique(y_true).size == 1:
            raise MetricComputeError("r2 undefined: all y_true values are identical (zero variance).")
        return _result_metric(float(r2_score(y_true, y_pred)), spec)


@register_metric("mape")
class MAPEMetric:
    """Mean Absolute Percentage Error for regression models.

    Expressed as a percentage (e.g. 5.0 means 5 %).  Undefined when any
    true value is zero; raises InsufficientDataError when all are zero.
    """

    name = "mape"
    metric_type = MetricType.performance
    requires_reference = False

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute MAPE, ignoring zero true-value rows.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: TabularSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value >= 0 (percentage).

        Raises:
            InsufficientDataError: If every y_true value is zero or near-zero
                (``|y_true| <= eps``).
        """
        s = _check_tabular(schema)
        y_true, y_pred, _ = _extract_arrays(current, s)
        eps = spec.params.get("eps", 1e-8)
        mask = np.abs(y_true) > eps
        if not mask.any():
            raise InsufficientDataError("MAPE undefined: all y_true values are zero or near-zero.")
        dropped = int((~mask).sum())
        if dropped:
            _logger.warning("mape: %d row(s) with |y_true| <= eps (%.2e) excluded.", dropped, eps)
        value = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        return _result_metric(value, spec)
