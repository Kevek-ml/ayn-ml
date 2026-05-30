"""Non-parametric two-sample statistical tests for tabular features.

P-value tests: a low p-value indicates drift between the two distributions.
Distance tests (hellinger, jensenshannon, tvd, energy_distance): a high value
indicates drift — use an upper-bound threshold.

Numeric tests:
- KS 2-sample        — omnibus CDF test (p-value).
- T-test             — mean-equality test, parametric (p-value).
- Mann-Whitney U     — non-parametric mean-rank test (p-value).
- Levene             — variance-equality test (p-value).
- Cramér-von Mises   — two-sample CDF test (p-value).
- Anderson-Darling   — tail-sensitive k-sample test (p-value).
- Epps-Singleton     — frequency-domain test (p-value).
- Hellinger          — histogram-based symmetric distance [0, 1].
- Jensen-Shannon     — symmetric KL-divergence distance [0, 1].
- TVD                — total variation distance [0, 1].
- Energy distance    — energy statistics distance [0, ∞).

Categorical tests:
- Chi-square         — homogeneity test; Cramér's V as effect size (p-value).
- G-test             — log-likelihood ratio; Cramér's V as effect size (p-value).
- Fisher exact       — exact 2×2 test for binary columns (p-value).
- Hellinger / JS / TVD — also accept categorical columns.

Binary-only:
- Z-test proportions — two-proportion z-test; z-score as effect size (p-value).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import ColumnType, DataSchema  # noqa: F401 — kept for public type annotation
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import SchemaError
from ayn_ml.metrics.registry import register_metric
from ayn_ml.metrics.tabular._helpers import _require_reference, _result_metric, extract_feature

_logger = logging.getLogger(__name__)
_MIN_ROWS = 10
"""Minimum rows per array required for a valid statistical test."""


def _extract_float_strict(df: Any, feature_name: str | None) -> np.ndarray:
    """Extract a feature column as a float64 array, enforcing the minimum-row threshold.

    Args:
        df: Input DataFrame (any narwhals-compatible frame).
        feature_name: Column to extract.  Must not be ``None``.

    Returns:
        1-D float64 numpy array with at least ``_MIN_ROWS`` elements.

    Raises:
        SchemaError: If ``feature_name`` is ``None`` or the column is absent.
        InsufficientDataError: If the array has fewer than ``_MIN_ROWS`` rows.
    """
    return extract_feature(df, feature_name, as_float=True, min_rows=_MIN_ROWS)


def _extract_native_strict(df: Any, feature_name: str | None) -> np.ndarray:
    """Extract a feature column preserving its native dtype, enforcing the minimum-row threshold.

    Used for categorical and binary features where dtype carries meaning.

    Args:
        df: Input DataFrame (any narwhals-compatible frame).
        feature_name: Column to extract.  Must not be ``None``.

    Returns:
        1-D numpy array with the column's native dtype and at least
        ``_MIN_ROWS`` elements.

    Raises:
        SchemaError: If ``feature_name`` is ``None`` or the column is absent.
        InsufficientDataError: If the array has fewer than ``_MIN_ROWS`` rows.
    """
    return extract_feature(df, feature_name, as_float=False, min_rows=_MIN_ROWS)


def _to_probs(ref: np.ndarray, cur: np.ndarray, n_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Normalize two arrays to probability distributions over a shared support.

    For numeric arrays builds a common histogram; for categorical arrays counts
    occurrences over the union of observed categories.

    Args:
        ref: Reference array.
        cur: Current array.
        n_bins: Histogram bin count for numeric data.  Adaptive when ``None``:
            ``min(max(sqrt(n), 10), 100)``.

    Returns:
        Pair of normalized probability arrays summing to 1.
    """
    if np.issubdtype(ref.dtype, np.number) and np.issubdtype(cur.dtype, np.number):
        combined = np.concatenate([ref, cur])
        if n_bins is None:
            n_bins = min(max(int(np.sqrt(len(combined))), 10), 100)
        edges = np.histogram_bin_edges(combined, bins=n_bins)
        p_ref = np.histogram(ref, bins=edges)[0].astype(float)
        p_cur = np.histogram(cur, bins=edges)[0].astype(float)
    else:
        all_cats = np.union1d(np.unique(ref), np.unique(cur))
        p_ref = np.array([np.sum(ref == c) for c in all_cats], dtype=float)
        p_cur = np.array([np.sum(cur == c) for c in all_cats], dtype=float)
    ref_sum = p_ref.sum()
    cur_sum = p_cur.sum()
    if ref_sum > 0:
        p_ref = p_ref / ref_sum
    if cur_sum > 0:
        p_cur = p_cur / cur_sum
    return p_ref, p_cur


@register_metric("ks_2samp")
class KS2SampMetric:
    """Kolmogorov-Smirnov two-sample test.

    Detects any difference in the cumulative distribution functions of
    reference and current samples.  Sensitive to location, scale, and shape
    shifts.  Returns the two-sided p-value.
    """

    name = "ks_2samp"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the KS two-sample test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold=0.05`` + ``upper_bound=False``
                to alert when p-value < threshold (drift detected).

        Returns:
            MetricResult with p-value in [0, 1].

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import ks_2samp

        _require_reference(reference, "ks_2samp")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        statistic, pvalue = ks_2samp(ref, cur)
        return _result_metric(pvalue, spec, effect_size=statistic, effect_size_label="ks_statistic")


@register_metric("ttest")
class TTestMetric:
    """Welch's two-sample t-test for mean equality (default) or Student's.

    Parametric test; assumes approximately normally distributed data.
    Defaults to Welch's variant (``equal_var=False``) which is safe when
    variances differ.  Set ``spec.params["equal_var"]`` to ``True`` to use
    Student's t-test (equal-variance assumption).

    ``effect_size`` returns Cohen's d = (mean_ref - mean_cur) / pooled_std,
    where pooled_std uses the df-weighted formula
    ``sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))``.
    """

    name = "ttest"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the independent-samples t-test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["equal_var"]`` controls whether to
                assume equal variances (default ``False`` — Welch's t-test).

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` =
            Cohen's d.

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import ttest_ind

        _require_reference(reference, "ttest")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        _, pvalue = ttest_ind(ref, cur, equal_var=spec.params.get("equal_var", False))
        n1, n2 = len(ref), len(cur)
        pooled_var = ((n1 - 1) * ref.var(ddof=1) + (n2 - 1) * cur.var(ddof=1)) / (n1 + n2 - 2)
        pooled_std = float(np.sqrt(pooled_var))
        cohens_d = float((ref.mean() - cur.mean()) / pooled_std) if pooled_std > 0 else 0.0
        return _result_metric(pvalue, spec, effect_size=cohens_d, effect_size_label="cohen_d")


@register_metric("mannwhitney")
class MannWhitneyMetric:
    """Mann-Whitney U test (Wilcoxon rank-sum test).

    Non-parametric alternative to the t-test; compares rank distributions.
    Robust to non-normality.  ``spec.params["alternative"]`` controls the
    hypothesis direction (default ``"two-sided"``).

    ``effect_size`` returns Cliff's delta in [-1, 1]: proportion of
    reference–current pairs where reference > current minus the reverse.
    """

    name = "mannwhitney"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Mann-Whitney U test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["alternative"]`` controls the
                alternative hypothesis (``"two-sided"``, ``"less"``,
                ``"greater"``; default ``"two-sided"``).

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` =
            Cliff's delta.

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import mannwhitneyu

        _require_reference(reference, "mannwhitney")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        alternative = spec.params.get("alternative", "two-sided")
        u_stat, pvalue = mannwhitneyu(ref, cur, alternative=alternative)
        cliffs_delta = float(2 * u_stat / (len(ref) * len(cur)) - 1)
        return _result_metric(pvalue, spec, effect_size=cliffs_delta, effect_size_label="cliff_delta")


@register_metric("levene")
class LeveneMetric:
    """Levene's test for equality of variances.

    Detects heteroscedasticity (variance drift) between the reference and
    current windows.  Less sensitive to departures from normality than
    Bartlett's test.

    ``effect_size`` returns the variance ratio cur_var / ref_var.  Values
    far from 1.0 indicate meaningful scale drift regardless of the p-value.
    """

    name = "levene"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Levene test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` =
            variance ratio (cur / ref).

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import levene

        _require_reference(reference, "levene")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        _, pvalue = levene(ref, cur)
        ref_var = float(ref.var(ddof=1))
        variance_ratio = float(cur.var(ddof=1) / ref_var) if ref_var > 0 else None
        label = "variance_ratio" if variance_ratio is not None else None
        return _result_metric(pvalue, spec, effect_size=variance_ratio, effect_size_label=label)


@register_metric("cramervonmises")
class CramerVonMisesMetric:
    """Cramér-von Mises two-sample test.

    Detects differences in the empirical CDFs, with greater power than the
    KS test against certain alternatives (especially location shifts).
    Delegates to ``scipy.stats.cramervonmises_2samp``.
    """

    name = "cramervonmises"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Cramér-von Mises two-sample test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with p-value in [0, 1].

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import cramervonmises_2samp

        _require_reference(reference, "cramervonmises")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        result = cramervonmises_2samp(ref, cur)
        return _result_metric(result.pvalue, spec)


@register_metric("chisquare")
class ChiSquareMetric:
    """Chi-square test of homogeneity for categorical features.

    Tests whether reference and current windows were drawn from the same
    categorical distribution by building a 2×K contingency table (one row
    per window, one column per category) and applying Pearson's chi-square
    test.  Returns the two-sided p-value.

    ``effect_size`` returns Cramér's V in [0, 1]:

        V = sqrt(χ² / n)

    where n is the combined sample size.  For a 2×K table the general
    formula ``sqrt(χ² / (n × (min(r, c) − 1)))`` simplifies to
    ``sqrt(χ² / n)`` because ``min(2, K) − 1 = 1``.  For a binary feature
    (K = 2) this equals the phi coefficient.

    Interpretation of Cramér's V:
    - V < 0.1  — negligible
    - V < 0.3  — moderate
    - V ≥ 0.3  — strong

    Raises ``SchemaError`` if the column has a floating-point dtype — use
    ``ks_2samp`` or ``wasserstein`` for continuous numeric features.

    A warning is emitted when any cell in the expected frequency table is
    below 5 (chi-square approximation becomes unreliable).
    """

    name = "chisquare"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.categorical, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the chi-square test of homogeneity p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold=0.05`` + ``upper_bound=False``
                to alert when p-value < threshold (drift detected).

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` =
            Cramér's V in [0, 1].

        Raises:
            SchemaError: If ``reference`` is ``None``, ``feature_name`` is
                missing, the column is absent, or either window's column dtype
                is floating-point.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import chi2_contingency

        _require_reference(reference, "chisquare")
        ref = _extract_native_strict(reference, spec.feature_name)
        cur = _extract_native_strict(current, spec.feature_name)

        if np.issubdtype(ref.dtype, np.floating) or np.issubdtype(cur.dtype, np.floating):
            dtype = ref.dtype if np.issubdtype(ref.dtype, np.floating) else cur.dtype
            raise SchemaError(
                f"chisquare requires a categorical column ('{spec.feature_name}' has dtype "
                f"{dtype}). Use ks_2samp or wasserstein for continuous features."
            )

        all_cats = np.union1d(np.unique(ref), np.unique(cur))

        # Single-category edge case: distributions are trivially identical.
        if len(all_cats) < 2:
            return _result_metric(1.0, spec, effect_size=0.0, effect_size_label="cramer_v")

        ref_counts = np.array([np.sum(ref == c) for c in all_cats], dtype=float)
        cur_counts = np.array([np.sum(cur == c) for c in all_cats], dtype=float)
        table = np.vstack([ref_counts, cur_counts])

        chi2, pvalue, _, expected = chi2_contingency(table)

        low_expected = int(np.sum(expected < 5))
        if low_expected:
            _logger.warning(
                "chisquare '%s': %d cell(s) have expected frequency < 5. "
                "Consider merging rare categories for a more reliable test.",
                spec.feature_name,
                low_expected,
            )

        n = len(ref) + len(cur)
        cramer_v = float(np.sqrt(chi2 / n))
        return _result_metric(pvalue, spec, effect_size=cramer_v, effect_size_label="cramer_v")


@register_metric("hellinger")
class HellingerMetric:
    """Hellinger distance between reference and current distributions.

    Symmetric, bounded [0, 1]: 0 means identical distributions, 1 means
    fully disjoint.  Builds a shared histogram for numeric features; uses
    category counts for categorical features.  Higher values indicate more
    drift — use an upper-bound threshold.

    ``spec.params["bins"]`` overrides the adaptive bin count (min 10, max 100,
    default based on sqrt of combined sample size).
    """

    name = "hellinger"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset(
        {ColumnType.numeric, ColumnType.binary, ColumnType.categorical}
    )

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Hellinger distance.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold`` as an upper bound to alert
                when distance > threshold (drift detected).

        Returns:
            MetricResult with distance in [0, 1].

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        _require_reference(reference, "hellinger")
        ref = _extract_native_strict(reference, spec.feature_name)
        cur = _extract_native_strict(current, spec.feature_name)
        n_bins: int | None = spec.params.get("bins")
        p_ref, p_cur = _to_probs(ref, cur, n_bins)
        distance = float(np.sqrt(np.sum((np.sqrt(p_ref) - np.sqrt(p_cur)) ** 2)) / np.sqrt(2))
        return _result_metric(distance, spec)


@register_metric("jensenshannon")
class JensenShannonMetric:
    """Jensen-Shannon distance between reference and current distributions.

    Symmetric square root of the JS divergence, bounded [0, 1].  Based on
    ``scipy.spatial.distance.jensenshannon``.  Builds a shared histogram for
    numeric features; uses category counts for categorical features.  Higher
    values indicate more drift.

    ``spec.params["bins"]`` overrides the adaptive bin count.
    """

    name = "jensenshannon"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset(
        {ColumnType.numeric, ColumnType.binary, ColumnType.categorical}
    )

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Jensen-Shannon distance.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold`` as an upper bound to alert
                when distance > threshold.

        Returns:
            MetricResult with distance in [0, 1].

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.spatial.distance import jensenshannon

        _require_reference(reference, "jensenshannon")
        ref = _extract_native_strict(reference, spec.feature_name)
        cur = _extract_native_strict(current, spec.feature_name)
        n_bins: int | None = spec.params.get("bins")
        p_ref, p_cur = _to_probs(ref, cur, n_bins)
        distance = float(jensenshannon(p_ref, p_cur))
        return _result_metric(distance, spec)


@register_metric("tvd")
class TVDMetric:
    """Total Variation Distance between reference and current distributions.

    TVD = 0.5 × Σ|p_i − q_i|, bounded [0, 1].  Builds a shared histogram
    for numeric features; uses category counts for categorical features.
    Higher values indicate more drift.

    ``spec.params["bins"]`` overrides the adaptive bin count.
    """

    name = "tvd"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset(
        {ColumnType.numeric, ColumnType.binary, ColumnType.categorical}
    )

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Total Variation Distance.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold`` as an upper bound to alert
                when distance > threshold.

        Returns:
            MetricResult with distance in [0, 1].

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        _require_reference(reference, "tvd")
        ref = _extract_native_strict(reference, spec.feature_name)
        cur = _extract_native_strict(current, spec.feature_name)
        n_bins: int | None = spec.params.get("bins")
        p_ref, p_cur = _to_probs(ref, cur, n_bins)
        distance = float(0.5 * np.sum(np.abs(p_ref - p_cur)))
        return _result_metric(distance, spec)


@register_metric("energy_distance")
class EnergyDistanceMetric:
    """Energy distance between reference and current numeric distributions.

    Based on ``scipy.stats.energy_distance``.  Returns a non-negative distance
    with no upper bound; 0 means identical distributions.  More sensitive to
    tail differences than the KS test.  Higher values indicate more drift.
    """

    name = "energy_distance"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the energy distance.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold`` as an upper bound to alert
                when distance > threshold.

        Returns:
            MetricResult with a non-negative distance value.

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import energy_distance

        _require_reference(reference, "energy_distance")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        return _result_metric(float(energy_distance(ref, cur)), spec)


@register_metric("anderson_darling")
class AndersonDarlingMetric:
    """Anderson-Darling k-sample test.

    More sensitive to differences in the tails than the KS test.  Uses
    ``scipy.stats.anderson_ksamp``.  Returns a p-value (scipy approximation
    clamps to [0.001, 0.25]).

    ``effect_size`` returns the AD test statistic.
    """

    name = "anderson_darling"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Anderson-Darling k-sample test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold=0.05`` + ``upper_bound=False``
                to alert when p-value < threshold (drift detected).

        Returns:
            MetricResult with p-value (clamped to [0.001, 0.25] by scipy's
            approximation) and ``effect_size`` = AD test statistic.

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import anderson_ksamp

        _require_reference(reference, "anderson_darling")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        result = anderson_ksamp([ref, cur])
        # .pvalue added in scipy 1.8; fall back to .significance_level for older versions
        pvalue = float(getattr(result, "pvalue", result.significance_level))
        return _result_metric(pvalue, spec, effect_size=float(result.statistic), effect_size_label="ad_statistic")


@register_metric("epps_singleton")
class EppsSingletonMetric:
    """Epps-Singleton two-sample test.

    Frequency-domain test based on the empirical characteristic function.
    Generally more powerful than KS for alternatives where the CDFs are close
    but the characteristic functions differ (e.g. different shapes).
    Delegates to ``scipy.stats.epps_singleton_2samp``.

    ``effect_size`` returns the test statistic (chi-square distributed under
    H₀ with 25 degrees of freedom).
    """

    name = "epps_singleton"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the Epps-Singleton two-sample test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold=0.05`` + ``upper_bound=False``
                to alert when p-value < threshold.

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` = ES
            test statistic.

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import epps_singleton_2samp

        _require_reference(reference, "epps_singleton")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        statistic, pvalue = epps_singleton_2samp(ref, cur)
        return _result_metric(float(pvalue), spec, effect_size=float(statistic), effect_size_label="es_statistic")


@register_metric("fisher_exact")
class FisherExactMetric:
    """Fisher's exact test for binary 2×2 contingency tables.

    Exact test for association between two binary outcomes across the reference
    and current windows.  Preferred over chi-square when any expected cell
    frequency is below 5.

    ``effect_size`` returns the odds ratio.

    ``spec.params["alternative"]`` controls the alternative hypothesis:
    ``"two-sided"`` (default), ``"less"``, or ``"greater"``.
    """

    name = "fisher_exact"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute Fisher's exact test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["alternative"]`` selects the
                alternative hypothesis (default ``"two-sided"``).

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` =
            odds ratio.

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import fisher_exact

        _require_reference(reference, "fisher_exact")
        ref = _extract_native_strict(reference, spec.feature_name)
        cur = _extract_native_strict(current, spec.feature_name)
        ref_pos = int(np.sum(ref != 0))
        cur_pos = int(np.sum(cur != 0))
        table = [[ref_pos, len(ref) - ref_pos], [cur_pos, len(cur) - cur_pos]]
        alternative = spec.params.get("alternative", "two-sided")
        odds_ratio, pvalue = fisher_exact(table, alternative=alternative)
        return _result_metric(float(pvalue), spec, effect_size=float(odds_ratio), effect_size_label="odds_ratio")


@register_metric("gtest")
class GTestMetric:
    """G-test (log-likelihood ratio) for categorical homogeneity.

    Tests whether reference and current windows were drawn from the same
    categorical distribution using the G-statistic rather than Pearson's χ².
    Preferred when expected frequencies are small or when the sample is sparse.

    ``effect_size`` returns Cramér's V in [0, 1].

    A warning is emitted when any expected cell frequency is below 5.

    Raises ``SchemaError`` for floating-point columns; use ``ks_2samp`` or
    a distance metric for continuous features.
    """

    name = "gtest"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.categorical, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the G-test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold=0.05`` + ``upper_bound=False``
                to alert when p-value < threshold (drift detected).

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` =
            Cramér's V in [0, 1].

        Raises:
            SchemaError: If ``reference`` is ``None``, ``feature_name`` is
                missing, or the column dtype is floating-point.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import chi2_contingency

        _require_reference(reference, "gtest")
        ref = _extract_native_strict(reference, spec.feature_name)
        cur = _extract_native_strict(current, spec.feature_name)

        if np.issubdtype(ref.dtype, np.floating) or np.issubdtype(cur.dtype, np.floating):
            dtype = ref.dtype if np.issubdtype(ref.dtype, np.floating) else cur.dtype
            raise SchemaError(
                f"gtest requires a categorical column ('{spec.feature_name}' has dtype "
                f"{dtype}). Use ks_2samp or a distance metric for continuous features."
            )

        all_cats = np.union1d(np.unique(ref), np.unique(cur))
        if len(all_cats) < 2:
            return _result_metric(1.0, spec, effect_size=0.0, effect_size_label="cramer_v")

        ref_counts = np.array([np.sum(ref == c) for c in all_cats], dtype=float)
        cur_counts = np.array([np.sum(cur == c) for c in all_cats], dtype=float)
        table = np.vstack([ref_counts, cur_counts])
        g_stat, pvalue, _, expected = chi2_contingency(table, lambda_="log-likelihood")

        low_expected = int(np.sum(expected < 5))
        if low_expected:
            _logger.warning(
                "gtest '%s': %d cell(s) have expected frequency < 5. "
                "Consider merging rare categories for a more reliable test.",
                spec.feature_name,
                low_expected,
            )

        n = len(ref) + len(cur)
        cramer_v = float(np.sqrt(g_stat / n)) if g_stat >= 0 else 0.0
        return _result_metric(pvalue, spec, effect_size=cramer_v, effect_size_label="cramer_v")


@register_metric("ztest_proportions")
class ZTestProportionsMetric:
    """Two-proportion z-test for binary features.

    Tests whether the proportion of positive values (1s) differs between
    reference and current windows using the pooled-proportion z-test.
    Assumes large samples (n ≥ 30); for small samples use ``fisher_exact``.

    ``effect_size`` returns the absolute z-score.
    """

    name = "ztest_proportions"
    metric_type = MetricType.drift
    requires_reference = True
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the two-proportion z-test p-value.

        Args:
            current: Current-window DataFrame.
            reference: Reference-window DataFrame.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; set ``threshold=0.05`` + ``upper_bound=False``
                to alert when p-value < threshold.

        Returns:
            MetricResult with p-value in [0, 1] and ``effect_size`` =
            absolute z-score.

        Raises:
            SchemaError: If ``reference`` is ``None`` or column is missing.
            InsufficientDataError: If either array has fewer than
                ``_MIN_ROWS`` rows.
        """
        from scipy.stats import norm

        _require_reference(reference, "ztest_proportions")
        ref = _extract_float_strict(reference, spec.feature_name)
        cur = _extract_float_strict(current, spec.feature_name)
        n1, n2 = len(ref), len(cur)
        p1 = float(np.mean(ref > 0.5))
        p2 = float(np.mean(cur > 0.5))
        p_hat = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = float(np.sqrt(p_hat * (1.0 - p_hat) * (1.0 / n1 + 1.0 / n2)))
        if se == 0.0:
            return _result_metric(1.0, spec)
        z = (p1 - p2) / se
        pvalue = float(2.0 * (1.0 - norm.cdf(abs(z))))
        return _result_metric(pvalue, spec, effect_size=abs(z), effect_size_label="z_score")
