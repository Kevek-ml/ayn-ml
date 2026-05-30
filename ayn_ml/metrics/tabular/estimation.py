"""Confidence-Based Performance Estimation (CBPE) for tabular models.

Estimates classification performance metrics on unlabelled windows by treating
calibrated predicted probabilities as probabilistic targets.  No ground-truth
labels are needed on the current window — the reference window (which has
labels) is used only to fit the probability calibrator.

Core idea: if a model assigns p̂ = 0.9 to an observation, there is a 90%
chance that observation is a true positive.  Summing these fractional
certainties across all observations reconstructs an estimated confusion matrix
and the derived metrics (accuracy, F1, precision, recall, AUC).

Theoretical basis: Vandewiele et al. (NannyML, 2022).

Supported metrics (all prefixed ``cbpe_``):

=================  ===============================================================
``cbpe_accuracy``  Estimated accuracy from fractional correct predictions.
``cbpe_auc``       Estimated ROC AUC by sweeping thresholds over p̂.
``cbpe_f1``        Estimated F1 from fractional TP / FP / FN (binary only).
``cbpe_precision`` Estimated precision from fractional TP / FP (binary only).
``cbpe_recall``    Estimated recall from fractional TP / FN (binary only).
=================  ===============================================================

All metrics set ``requires_reference = True`` (reference provides labels for
the calibrator) and ``metric_type = MetricType.performance``.

Schema requirements:
    ``TabularSchema.proba_col`` must be set and present on both windows.
    ``TabularSchema.prediction_col`` is required for accuracy, F1, precision,
    and recall.  ``TabularSchema.label_col`` is required on the reference
    window only.

Key limitations:
    **Blind to concept drift** (P(Y|X) shift): if the true label-generating
    process changes while the model's probabilities remain stable, CBPE
    reports no performance change.  Always pair with a drift metric (PSI,
    Wasserstein) on ``y_pred_proba`` to detect this failure mode.

    Binary classification only for accuracy, F1, precision, and recall.
    AUC supports multi-class when ``spec.params["multi_class"]`` is set, but
    CBPE AUC in multi-class mode is not yet validated — use with caution.

    Estimates degrade significantly below 300 observations.  The minimum
    enforced threshold is 100 rows; below that ``InsufficientDataError`` is
    raised.

    Calibration is applied by default (``spec.params["calibrate"] = True``).
    Disable with ``False`` only if your model is already well-calibrated.

Future work (not in v1):
    **DLE (Direct Loss Estimation)** for regression metrics: trains a LightGBM
    nanny model on ``(X_ref, loss_ref)`` to predict expected loss on unlabelled
    current.  Deferred because it requires feature columns
    (``TabularSchema.feature_cols``), a heavy optional dependency
    (``lightgbm``), and a stateful fit/estimate lifecycle that does not map
    cleanly to the stateless ``Metric`` protocol.

    **DiagnosticReport**: cross-metric pattern matching to produce
    human-readable verdicts over a ``MonitoringReport`` (e.g. "CBPE stable +
    PSI high → covariate shift without performance impact").  Planned as a
    standalone utility once the metric library is more complete.
"""

from __future__ import annotations

import logging
from typing import Any

import narwhals as nw
import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import DataSchema
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import InsufficientDataError, MetricComputeError, SchemaError
from ayn_ml.metrics.registry import register_metric
from ayn_ml.metrics.tabular._helpers import _check_tabular, _extract_pred, _extract_proba, _result_metric

_logger = logging.getLogger(__name__)

_MIN_ROWS = 100
"""Minimum rows in the current window for a reliable CBPE estimate."""

_MIN_REF_ROWS = 100
"""Minimum reference rows needed to fit the isotonic calibrator."""


def _check_size(n: int, minimum: int, window: str) -> None:
    """Raise InsufficientDataError when a window has too few rows.

    Args:
        n: Actual row count.
        minimum: Required minimum.
        window: Human-readable label for the error message.

    Raises:
        InsufficientDataError: If ``n < minimum``.
    """
    if n < minimum:
        raise InsufficientDataError(f"CBPE requires at least {minimum} rows in the {window} window; got {n}.")


# ── Calibration ────────────────────────────────────────────────────────────────


def _calibrate(
    y_true_ref: np.ndarray,
    y_proba_ref: np.ndarray,
    y_proba_cur: np.ndarray,
) -> np.ndarray:
    """Fit an isotonic calibrator on the reference window and apply it to current.

    Isotonic regression is a non-parametric monotone calibration method.
    Out-of-bound current values are clipped to ``[0, 1]``.

    Args:
        y_true_ref: Binary ground-truth labels from the reference window.
        y_proba_ref: Raw predicted probabilities from the reference window.
        y_proba_cur: Raw predicted probabilities from the current window.

    Returns:
        Calibrated probabilities for the current window, clipped to ``[0, 1]``.
    """
    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_proba_ref, y_true_ref)
    return calibrator.predict(y_proba_cur)


# ── Shared setup ───────────────────────────────────────────────────────────────


def _setup_cbpe(
    current: Any,
    reference: Any,
    schema: DataSchema,
    spec: MetricSpec,
    need_pred: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Shared extraction and calibration logic for all CBPE metrics.

    Extracts probabilities from both windows, validates sizes, optionally fits
    an isotonic calibrator on the reference, and returns calibrated current
    probabilities.

    Args:
        current: Current-window DataFrame (no ``y_true``).
        reference: Reference-window DataFrame (must contain ``label_col``).
        schema: TabularSchema describing column names.
        spec: MetricSpec; ``params["calibrate"]`` controls calibration
            (default ``True``).
        need_pred: When ``True``, also extracts ``y_pred`` from the current
            window (required for accuracy, F1, precision, recall).

    Returns:
        A two-tuple ``(y_pred, p_hat)`` where ``y_pred`` is ``None`` when
        ``need_pred=False`` and ``p_hat`` is the calibrated probability array
        for the current window.

    Raises:
        SchemaError: If reference is ``None``, or required columns are absent.
        InsufficientDataError: If either window has fewer rows than the minimum.
    """
    if reference is None:
        raise SchemaError("CBPE metrics require a reference window for calibration.")

    s = _check_tabular(schema)

    y_proba_cur = _extract_proba(current, s, "current")
    _check_size(len(y_proba_cur), _MIN_ROWS, "current")

    y_proba_ref = _extract_proba(reference, s, "reference")
    _check_size(len(y_proba_ref), _MIN_REF_ROWS, "reference")

    ref_native = nw.from_native(reference, eager_only=True)
    if s.label_col not in ref_native.columns:
        raise SchemaError(f"Column '{s.label_col}' not found in reference window.")
    y_true_ref = ref_native[s.label_col].to_numpy().astype(float)

    if spec.params.get("calibrate", True):
        p_hat = _calibrate(y_true_ref, y_proba_ref, y_proba_cur)
    else:
        p_hat = np.clip(y_proba_cur, 0.0, 1.0)

    y_pred = _extract_pred(current, s) if need_pred else None
    return y_pred, p_hat


# ── Estimation math ────────────────────────────────────────────────────────────


def _estimate_confusion(
    y_pred: np.ndarray,
    p_hat: np.ndarray,
) -> tuple[float, float, float, float]:
    """Estimate fractional TP, FP, FN, TN from calibrated probabilities.

    For each observation i:

    * If predicted positive (y_pred > 0.5): contributes p̂ᵢ to TP and
      (1 − p̂ᵢ) to FP.
    * If predicted negative: contributes p̂ᵢ to FN and (1 − p̂ᵢ) to TN.

    Args:
        y_pred: Hard binary predictions (0 / 1).
        p_hat: Calibrated predicted probabilities for the positive class.

    Returns:
        Four-tuple ``(tp, fp, fn, tn)`` of estimated confusion matrix entries.
    """
    pos = y_pred > 0.5
    tp = float(np.sum(p_hat[pos]))
    fp = float(np.sum(1.0 - p_hat[pos]))
    fn = float(np.sum(p_hat[~pos]))
    tn = float(np.sum(1.0 - p_hat[~pos]))
    return tp, fp, fn, tn


def _estimate_accuracy(y_pred: np.ndarray, p_hat: np.ndarray) -> float:
    """Estimate accuracy as the mean probability of being correct.

    For each predicted positive: contributes p̂ (probability it is a true
    positive).  For each predicted negative: contributes (1 − p̂) (probability
    it is a true negative).

    Args:
        y_pred: Hard binary predictions (0 / 1).
        p_hat: Calibrated probabilities for the positive class.

    Returns:
        Estimated accuracy in [0, 1].
    """
    pos = (y_pred > 0.5).astype(float)
    return float(np.mean(p_hat * pos + (1.0 - p_hat) * (1.0 - pos)))


def _estimate_auc(p_hat: np.ndarray) -> float:
    """Estimate ROC AUC by sweeping thresholds over calibrated probabilities.

    At threshold t, observations with p̂ ≥ t are predicted positive:

    * ``TPR(t) = Σ p̂[p̂≥t] / Σ p̂``
    * ``FPR(t) = Σ (1−p̂)[p̂≥t] / Σ (1−p̂)``

    The ROC curve is traced by sorting observations by descending p̂ and
    accumulating fractional TP and FP.  Area is computed via the trapezoidal
    rule after sorting by FPR to handle any ties.

    Args:
        p_hat: Calibrated probabilities for the positive class.

    Returns:
        Estimated AUC in [0, 1].

    Raises:
        MetricComputeError: If estimated class totals are near-zero (window
            appears effectively single-class).
    """
    total_pos = float(p_hat.sum())
    total_neg = float((1.0 - p_hat).sum())
    if total_pos < 1e-8 or total_neg < 1e-8:
        raise MetricComputeError(
            "cbpe_auc: estimated class totals near-zero — window appears effectively single-class."
        )
    desc_idx = np.argsort(-p_hat)
    p_sorted = p_hat[desc_idx]
    tpr = np.concatenate([[0.0], np.cumsum(p_sorted) / total_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - p_sorted) / total_neg])
    sort_idx = np.argsort(fpr)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(tpr[sort_idx], fpr[sort_idx]))


# ── Metric classes ─────────────────────────────────────────────────────────────


@register_metric("cbpe_accuracy")
class CBPEAccuracyMetric:
    """Estimated accuracy using Confidence-Based Performance Estimation.

    Does not require ground-truth labels on the current window.  See module
    docstring for assumptions and limitations.
    """

    name = "cbpe_accuracy"
    metric_type = MetricType.performance
    requires_reference = True

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Estimate accuracy from calibrated probabilities.

        Args:
            current: Current-window DataFrame (no ``y_true`` required).
            reference: Reference-window DataFrame (requires ``label_col``).
            schema: TabularSchema with ``proba_col`` and ``prediction_col``.
            spec: MetricSpec; ``params["calibrate"]`` (default ``True``)
                toggles isotonic calibration.

        Returns:
            MetricResult with estimated accuracy in [0, 1].

        Raises:
            SchemaError: If reference is ``None`` or required columns absent.
            InsufficientDataError: If either window has fewer than
                ``_MIN_ROWS`` rows.
        """
        y_pred, p_hat = _setup_cbpe(current, reference, schema, spec, need_pred=True)
        return _result_metric(_estimate_accuracy(y_pred, p_hat), spec)


@register_metric("cbpe_auc")
class CBPEAUCMetric:
    """Estimated ROC AUC using Confidence-Based Performance Estimation.

    Does not require ground-truth labels on the current window.  See module
    docstring for assumptions and limitations.
    """

    name = "cbpe_auc"
    metric_type = MetricType.performance
    requires_reference = True

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Estimate ROC AUC from calibrated probabilities.

        Args:
            current: Current-window DataFrame (no ``y_true`` required).
            reference: Reference-window DataFrame (requires ``label_col``).
            schema: TabularSchema with ``proba_col`` set.
            spec: MetricSpec; ``params["calibrate"]`` (default ``True``)
                toggles isotonic calibration.

        Returns:
            MetricResult with estimated AUC in [0, 1].

        Raises:
            SchemaError: If reference is ``None`` or required columns absent.
            InsufficientDataError: If either window has fewer than
                ``_MIN_ROWS`` rows.
            MetricComputeError: If the window appears single-class.
        """
        _, p_hat = _setup_cbpe(current, reference, schema, spec, need_pred=False)
        return _result_metric(_estimate_auc(p_hat), spec)


@register_metric("cbpe_f1")
class CBPEF1Metric:
    """Estimated F1 score using Confidence-Based Performance Estimation.

    Binary classification only.  Does not require ground-truth labels on the
    current window.  See module docstring for assumptions and limitations.
    """

    name = "cbpe_f1"
    metric_type = MetricType.performance
    requires_reference = True

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Estimate F1 score from calibrated probabilities.

        Args:
            current: Current-window DataFrame (no ``y_true`` required).
            reference: Reference-window DataFrame (requires ``label_col``).
            schema: TabularSchema with ``proba_col`` and ``prediction_col``.
            spec: MetricSpec; ``params["calibrate"]`` (default ``True``)
                toggles isotonic calibration.

        Returns:
            MetricResult with estimated F1 in [0, 1].

        Raises:
            SchemaError: If reference is ``None`` or required columns absent.
            InsufficientDataError: If either window has fewer than
                ``_MIN_ROWS`` rows.
            MetricComputeError: If estimated confusion matrix is near-zero.
        """
        y_pred, p_hat = _setup_cbpe(current, reference, schema, spec, need_pred=True)
        tp, fp, fn, _ = _estimate_confusion(y_pred, p_hat)
        denom = 2.0 * tp + fp + fn
        if denom < 1e-8:
            raise MetricComputeError("cbpe_f1: estimated confusion matrix is near-zero.")
        return _result_metric(2.0 * tp / denom, spec)


@register_metric("cbpe_precision")
class CBPEPrecisionMetric:
    """Estimated precision using Confidence-Based Performance Estimation.

    Binary classification only.  Does not require ground-truth labels on the
    current window.  See module docstring for assumptions and limitations.
    """

    name = "cbpe_precision"
    metric_type = MetricType.performance
    requires_reference = True

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Estimate precision from calibrated probabilities.

        Args:
            current: Current-window DataFrame (no ``y_true`` required).
            reference: Reference-window DataFrame (requires ``label_col``).
            schema: TabularSchema with ``proba_col`` and ``prediction_col``.
            spec: MetricSpec; ``params["calibrate"]`` (default ``True``)
                toggles isotonic calibration.

        Returns:
            MetricResult with estimated precision in [0, 1].

        Raises:
            SchemaError: If reference is ``None`` or required columns absent.
            InsufficientDataError: If either window has fewer than
                ``_MIN_ROWS`` rows.
            MetricComputeError: If there are no estimated predicted positives.
        """
        y_pred, p_hat = _setup_cbpe(current, reference, schema, spec, need_pred=True)
        tp, fp, _, _ = _estimate_confusion(y_pred, p_hat)
        denom = tp + fp
        if denom < 1e-8:
            raise MetricComputeError("cbpe_precision: no predicted positives in current window.")
        return _result_metric(tp / denom, spec)


@register_metric("cbpe_recall")
class CBPERecallMetric:
    """Estimated recall using Confidence-Based Performance Estimation.

    Binary classification only.  Does not require ground-truth labels on the
    current window.  See module docstring for assumptions and limitations.

    ``effect_size`` is not populated (CBPE produces a point estimate only).
    """

    name = "cbpe_recall"
    metric_type = MetricType.performance
    requires_reference = True

    def compute(
        self,
        current: Any,
        reference: Any | None,
        schema: DataSchema,
        spec: MetricSpec,
    ) -> MetricResult:
        """Estimate recall from calibrated probabilities.

        Args:
            current: Current-window DataFrame (no ``y_true`` required).
            reference: Reference-window DataFrame (requires ``label_col``).
            schema: TabularSchema with ``proba_col`` and ``prediction_col``.
            spec: MetricSpec; ``params["calibrate"]`` (default ``True``)
                toggles isotonic calibration.

        Returns:
            MetricResult with estimated recall in [0, 1].

        Raises:
            SchemaError: If reference is ``None`` or required columns absent.
            InsufficientDataError: If either window has fewer than
                ``_MIN_ROWS`` rows.
            MetricComputeError: If estimated total positives are near-zero.
        """
        y_pred, p_hat = _setup_cbpe(current, reference, schema, spec, need_pred=True)
        tp, _, fn, _ = _estimate_confusion(y_pred, p_hat)
        denom = tp + fn
        if denom < 1e-8:
            raise MetricComputeError("cbpe_recall: estimated total positives near-zero in current window.")
        return _result_metric(tp / denom, spec)
