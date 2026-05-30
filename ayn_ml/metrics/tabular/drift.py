"""Distribution-shift (drift) metrics for tabular features.

Provides four complementary drift detectors:

- PSI          — Population Stability Index; auto-detects numeric vs categorical columns.
- Wasserstein  — Earth Mover's Distance; numeric features only.
- MMD          — Maximum Mean Discrepancy with RBF kernel; numeric features only.
- target_drift — PSI on the label column; detects shifts in P(Y).

PSI uses histogram binning for numeric features and frequency counting for
categorical (string/object) features.  Wasserstein and MMD require numeric
columns and raise ``SchemaError`` otherwise.

PSI, Wasserstein, and MMD operate on a single feature specified via
``MetricSpec.feature_name``.  ``target_drift`` uses ``schema.label_col``
directly and does not require ``feature_name``.

All metrics require a reference window.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import ColumnType, DataSchema
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import InsufficientDataError, SchemaError
from ayn_ml.metrics.registry import register_metric
from ayn_ml.metrics.tabular._helpers import (
    _check_tabular,
    _extract_label,
    _is_numeric,
    _require_reference,
    _result_metric,
    extract_feature,
)

_logger = logging.getLogger(__name__)

_MIN_ROWS = 10
"""Minimum rows required per array for drift computation."""


def _check_sizes(*arrays: np.ndarray) -> None:
    """Assert that every array has at least ``_MIN_ROWS`` elements.

    Used for arrays extracted outside the ``extract_feature`` path (e.g.
    label and probability arrays in ``TargetDriftMetric``).

    Args:
        *arrays: One or more numpy arrays to validate.

    Raises:
        InsufficientDataError: If any array has fewer than ``_MIN_ROWS`` rows.
    """
    for arr in arrays:
        if len(arr) < _MIN_ROWS:
            raise InsufficientDataError(f"At least {_MIN_ROWS} rows required for drift computation.")


def _psi_core(
    ref_arr: np.ndarray,
    cur_arr: np.ndarray,
    spec: MetricSpec,
    col_name: str,
    *,
    is_numeric: bool,
    metric_name: str = "psi",
) -> float:
    """Compute PSI between two arrays.

    Shared by :class:`PSIMetric` and :class:`TargetDriftMetric` to avoid
    duplicating the histogram and frequency-count logic.

    Args:
        ref_arr: Reference array.
        cur_arr: Current array.
        spec: MetricSpec; reads ``params["n_bins"]`` (numeric, default 10) and
            ``params["eps"]`` (clipping floor; default ``1e-4`` numeric,
            ``1e-8`` categorical).
        col_name: Column name used in warning messages.
        is_numeric: If ``True``, use histogram binning; otherwise use
            frequency counts on the union of observed categories.
        metric_name: Metric name prefix for log messages (default ``"psi"``).

    Returns:
        PSI value (non-negative float).
    """
    if is_numeric:
        n_bins = int(spec.params.get("n_bins", 10))
        eps = float(spec.params.get("eps", 1e-4))
        ref_float = ref_arr.astype(float)
        cur_float = cur_arr.astype(float)
        all_float = np.concatenate([ref_float, cur_float])
        _, bin_edges = np.histogram(all_float, bins=n_bins)
        bin_edges[0] -= 1e-8
        bin_edges[-1] += 1e-8
        ref_min, ref_max = ref_float.min(), ref_float.max()
        out_of_range = int(np.sum((cur_float < ref_min) | (cur_float > ref_max)))
        if out_of_range:
            _logger.warning(
                "%s '%s': %d current value(s) outside the reference range "
                "[%.4g, %.4g]. PSI includes these values (union bins) but "
                "their contribution is sensitive to eps=%.4g — PSI varies "
                "%.1f–%.1f for that bin alone at 30%% out-of-range density. "
                "Cross-check with wasserstein or ks_2samp.",
                metric_name,
                col_name,
                out_of_range,
                ref_min,
                ref_max,
                eps,
                0.30 * np.log(0.30 / min(eps * 10, 0.30)),
                0.30 * np.log(0.30 / max(eps / 10, 1e-9)),
            )
        ref_counts, _ = np.histogram(ref_float, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_float, bins=bin_edges)
    else:
        eps = float(spec.params.get("eps", 1e-8))
        all_cats = np.union1d(np.unique(ref_arr), np.unique(cur_arr))
        ref_counts = np.array([np.sum(ref_arr == c) for c in all_cats], dtype=float)
        cur_counts = np.array([np.sum(cur_arr == c) for c in all_cats], dtype=float)

    ref_pct = np.clip(ref_counts / len(ref_arr), eps, None)
    cur_pct = np.clip(cur_counts / len(cur_arr), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@register_metric("psi")
class PSIMetric:
    """Population Stability Index (PSI) for numeric or categorical features.

    PSI measures how much a feature's distribution has shifted between a
    reference window and a current window:

        PSI = Σ (p_cur - p_ref) × ln(p_cur / p_ref)

    The column type is detected automatically at compute time:

    - **Numeric**: bin edges are derived from the union of reference and
      current values so no data points are silently dropped.
      ``params["n_bins"]`` controls resolution (default 10).
      ``params["eps"]`` controls the zero-clipping floor applied to both
      distributions before computing the log ratio (default ``1e-4``).
      When current values fall outside the reference range, those bins will
      have ``p_ref ≈ 0`` after clipping — their PSI contribution is
      mathematically valid but sensitive to the ``eps`` value.  A warning
      is emitted with the count of out-of-range values and a suggestion to
      cross-check with Wasserstein or ``ks_2samp``.
    - **Categorical**: bins are the union of all categories seen in either
      window; each category's frequency is used directly.

    Probabilities are clipped to ``params["eps"]`` to avoid log(0).  Default
    is ``1e-4`` for numeric features and ``1e-8`` for categorical features.

    Interpretation thresholds (rule of thumb):
    - PSI < 0.1  — no significant drift
    - PSI < 0.25 — moderate drift
    - PSI ≥ 0.25 — significant drift
    """

    name = "psi"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset(
        {ColumnType.numeric, ColumnType.categorical, ColumnType.binary}
    )

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute PSI between reference and current distributions.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["n_bins"]`` controls histogram
                resolution for numeric features (default 10); ignored for
                categorical features.  ``params["eps"]`` sets the clipping
                floor for zero probabilities (default ``1e-4``).

        Returns:
            MetricResult with value >= 0.

        Raises:
            SchemaError: If ``reference`` is ``None`` or ``feature_name``
                is missing/not found.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        _require_reference(reference, "PSI")

        feat = spec.feature_name
        reference_vals = extract_feature(reference, feat, as_float=False, min_rows=_MIN_ROWS)
        current_vals = extract_feature(current, feat, as_float=False, min_rows=_MIN_ROWS)

        psi = _psi_core(
            reference_vals,
            current_vals,
            spec,
            feat,
            is_numeric=_is_numeric(reference_vals, feat, schema),
        )
        return _result_metric(psi, spec)


@register_metric("wasserstein")
class WassersteinMetric:
    """Wasserstein-1 distance (Earth Mover's Distance) for a single feature.

    Measures the minimum "work" needed to transform the reference distribution
    into the current distribution.  Units match the feature's own units, making
    it easy to interpret (e.g. "the age distribution shifted by 3.4 years on
    average").

    Delegates to ``scipy.stats.wasserstein_distance``.
    """

    name = "wasserstein"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Wasserstein-1 distance.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value >= 0.

        Raises:
            SchemaError: If ``reference`` is ``None``, column is missing,
                or the column is non-numeric.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import wasserstein_distance

        _require_reference(reference, "Wasserstein")

        feat = spec.feature_name
        reference_vals = extract_feature(reference, feat, as_float=False, min_rows=_MIN_ROWS)
        current_vals = extract_feature(current, feat, as_float=False, min_rows=_MIN_ROWS)
        # Secondary guard for callers that bypass the Runner (e.g. direct compute() calls).
        # The Runner rejects categorical columns via accepted_column_types before reaching here.
        if not _is_numeric(reference_vals, feat, schema):
            raise SchemaError(f"Wasserstein requires a numeric feature ('{feat}' is {reference_vals.dtype}).")

        value = float(wasserstein_distance(reference_vals.astype(float), current_vals.astype(float)))
        return _result_metric(value, spec)


@register_metric("mmd")
class MMDMetric:
    """Maximum Mean Discrepancy (MMD) with an RBF kernel.

    MMD is a kernel-based two-sample test statistic.  It detects subtle
    shape differences that histogram-based measures can miss.  The kernel
    bandwidth (sigma) is estimated via the median heuristic on the reference
    data.

    O(n²) complexity is controlled via ``spec.params["max_samples"]``
    (default 500): both arrays are subsampled uniformly before the kernel
    matrix is computed.
    """

    name = "mmd"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the MMD between reference and current distributions.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["max_samples"]`` caps the subsample
                size (default 500) to bound O(n²) kernel computation.
                ``params["random_state"]`` seeds the subsampling RNG for
                reproducibility (default ``42``).

        Returns:
            MetricResult with value >= 0 (square root of MMD²).

        Raises:
            SchemaError: If ``reference`` is ``None``, column is missing,
                or the column is non-numeric.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        _require_reference(reference, "MMD")

        feat = spec.feature_name
        x = extract_feature(reference, feat, as_float=False, min_rows=_MIN_ROWS)
        y = extract_feature(current, feat, as_float=False, min_rows=_MIN_ROWS)
        # Secondary guard for callers that bypass the Runner (e.g. direct compute() calls).
        # The Runner rejects categorical columns via accepted_column_types before reaching here.
        if not _is_numeric(x, feat, schema):
            raise SchemaError(f"MMD requires a numeric feature ('{feat}' is {x.dtype}).")
        x = x.astype(float).reshape(-1, 1)
        y = y.astype(float).reshape(-1, 1)

        max_samples = spec.params.get("max_samples", 500)
        rng = np.random.default_rng(spec.params.get("random_state", 42))
        if len(x) > max_samples:
            x = x[rng.choice(len(x), max_samples, replace=False)]
        if len(y) > max_samples:
            y = y[rng.choice(len(y), max_samples, replace=False)]

        diffs = np.abs(x - x.T)
        sigma = float(np.median(diffs[np.triu_indices(len(x), k=1)])) or 1.0

        def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute the RBF (Gaussian) kernel matrix between a and b.

            Args:
                a: Array of shape (n, 1).
                b: Array of shape (m, 1).

            Returns:
                Kernel matrix of shape (n, m).
            """
            sq = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
            return np.exp(-sq / (2 * sigma**2))

        mmd2 = rbf(x, x).mean() - 2 * rbf(x, y).mean() + rbf(y, y).mean()
        return _result_metric(float(max(mmd2, 0.0)) ** 0.5, spec)


@register_metric("target_drift")
class TargetDriftMetric:
    """PSI on the label column — detects shifts in P(Y).

    Measures how much the target distribution has changed between the
    reference and current windows.  Uses the same PSI formula as
    :class:`PSIMetric` but reads ``schema.label_col`` directly, so
    ``spec.feature_name`` is not required.

    Target type is inferred from the column's dtype:

    - **Integer or string labels** (classification): treated as categorical;
      PSI is computed on class frequencies.
    - **Float labels** (regression): treated as numeric; PSI uses histogram
      binning (``params["n_bins"]``, default 10).

    Override with ``params["treat_as"]``:

    - ``"auto"``        — dtype-based inference (default)
    - ``"categorical"`` — force frequency-count PSI
    - ``"numeric"``     — force histogram-binned PSI

    Both windows must contain ``schema.label_col`` — in production this means
    the metric is computed when delayed labels become available.

    **Known limitation:** target drift is insensitive to symmetric concept
    drifts.  When P(X) is fixed and the drift is symmetric around the class
    boundary (e.g. boundary inversion or balanced feature-weight shift),
    P(Y) remains ≈ 0.5 and PSI ≈ 0 even though P(Y|X) has changed
    everywhere.  Pair with CBPE and covariate drift metrics for full coverage.
    """

    name = "target_drift"
    metric_type = MetricType.drift
    requires_reference = True

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute PSI on the label column between reference and current.

        Args:
            current: Current-window DataFrame (must contain ``schema.label_col``).
            reference: Reference-window DataFrame (must contain ``schema.label_col``).
            schema: TabularSchema with ``label_col`` set.
            spec: MetricSpec; ``params["n_bins"]`` controls histogram resolution
                for numeric targets (default 10).  ``params["eps"]`` sets the
                zero-clipping floor (default ``1e-4`` numeric, ``1e-8``
                categorical).  ``params["treat_as"]`` overrides dtype inference
                (``"auto"`` | ``"numeric"`` | ``"categorical"``).

        Returns:
            MetricResult with value >= 0.

        Raises:
            SchemaError: If ``reference`` is ``None``, ``schema`` is not a
                ``TabularSchema``, or ``label_col`` is absent from either window.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        _require_reference(reference, "target_drift")

        s = _check_tabular(schema)
        ref_arr = _extract_label(reference, s, "reference")
        cur_arr = _extract_label(current, s, "current")
        _check_sizes(ref_arr, cur_arr)

        treat_as = spec.params.get("treat_as", "auto")
        if treat_as == "numeric":
            use_numeric = True
        elif treat_as == "categorical":
            use_numeric = False
        elif treat_as == "auto":
            # float dtype → regression target; int/str → classification labels.
            # Intentionally np.floating (not np.number) so integer-encoded
            # classification labels default to categorical PSI.
            use_numeric = bool(np.issubdtype(ref_arr.dtype, np.floating))
        else:
            _logger.warning(
                "target_drift: unknown treat_as=%r; falling back to 'auto'. "
                "Valid values: 'auto', 'numeric', 'categorical'.",
                treat_as,
            )
            use_numeric = bool(np.issubdtype(ref_arr.dtype, np.floating))

        psi = _psi_core(ref_arr, cur_arr, spec, s.label_col, is_numeric=use_numeric, metric_name="target_drift")
        return _result_metric(psi, spec)
