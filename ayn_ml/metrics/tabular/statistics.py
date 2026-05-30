"""Descriptive statistics metrics for individual tabular features and datasets.

These metrics do not require a reference window.  They describe the
distribution of individual features or dataset-level properties and are used
for data-quality monitoring and basic profile dashboards.

Column-level metrics: mean, median, std, skewness, kurtosis, quantile, count,
top_category, sum, unique_count, in_range_count, out_range_count, in_list_count.

Dataset-level metrics (feature_name=None): row_count, column_count,
almost_constant_columns, duplicate_rows, empty_columns.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import ColumnType, DataSchema  # noqa: F401 — kept for public type annotation
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.metrics.registry import register_metric
from ayn_ml.metrics.tabular._helpers import _result_stat, extract_feature


@register_metric("mean")
class MeanMetric:
    """Arithmetic mean of a numeric feature."""

    name = "mean"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the mean.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a float value.
        """
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=1).astype(float)
        return _result_stat(float(np.mean(arr)), spec)


@register_metric("median")
class MedianMetric:
    """Median (50th percentile) of a numeric feature."""

    name = "median"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the median.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a float value.
        """
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=1).astype(float)
        return _result_stat(float(np.median(arr)), spec)


@register_metric("std")
class StdMetric:
    """Sample standard deviation of a numeric feature (ddof=1)."""

    name = "std"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the sample standard deviation.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with value >= 0.
        """
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=1).astype(float)
        return _result_stat(float(np.std(arr, ddof=1)), spec)


@register_metric("skewness")
class SkewnessMetric:
    """Fisher-Pearson standardised skewness of a numeric feature.

    Delegates to ``scipy.stats.skew``.  Positive skew = right tail is longer.
    """

    name = "skewness"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute skewness.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a finite float value.
        """
        from scipy.stats import skew

        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=1).astype(float)
        return _result_stat(float(skew(arr)), spec)


@register_metric("kurtosis")
class KurtosisMetric:
    """Excess kurtosis of a numeric feature.

    Delegates to ``scipy.stats.kurtosis`` (Fisher's definition, excess
    relative to the normal distribution).  0 = normal, > 0 = heavy tails.
    """

    name = "kurtosis"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute excess kurtosis.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a finite float value.
        """
        from scipy.stats import kurtosis

        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=1).astype(float)
        return _result_stat(float(kurtosis(arr)), spec)


@register_metric("quantile")
class QuantileMetric:
    """Arbitrary quantile of a numeric feature.

    The quantile probability is read from ``spec.params["q"]`` (default 0.5,
    i.e. the median).  Pass e.g. ``params={"q": 0.95}`` for the 95th
    percentile.
    """

    name = "quantile"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute a quantile.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["q"]`` sets the quantile probability
                in [0, 1] (default 0.5).

        Returns:
            MetricResult with a float value.
        """
        q = spec.params.get("q", 0.5)
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=1).astype(float)
        return _result_stat(float(np.quantile(arr, q)), spec)


@register_metric("count")
class CountMetric:
    """Total row count for a given feature column (including null values).

    Returns 0 (not an error) when the current window is empty, making it
    safe to use as a data-volume guard in monitoring pipelines.
    """

    name = "count"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset(
        {ColumnType.numeric, ColumnType.categorical, ColumnType.binary}
    )

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the row count.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with an integer value (0 when the window is empty).
        """
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=0)
        return _result_stat(int(len(arr)), spec)


@register_metric("top_category")
class TopCategoryMetric:
    """Most frequent category value (mode) in a categorical feature.

    Returns the category label as a string.  ``MetricResult.status`` is
    always ``None`` because string values cannot be compared to a numeric
    threshold.

    Tie-breaking: when two or more categories share the highest count,
    ``np.unique`` returns them in lexicographic order and ``np.argmax``
    picks the first — i.e. the alphabetically earliest tied value.
    """

    name = "top_category"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.categorical, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the most frequent category.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; threshold is ignored for string output.

        Returns:
            MetricResult with a string value and ``status=None``.
        """
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=1)
        values, counts = np.unique(arr, return_counts=True)
        top = str(values[np.argmax(counts)])
        return _result_stat(top, spec)


@register_metric("sum")
class SumMetric:
    """Sum of all values in a numeric feature."""

    name = "sum"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the sum.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a float value.

        Raises:
            SchemaError: If ``spec.feature_name`` is ``None`` or absent from the DataFrame.
        """
        arr = extract_feature(current, spec.feature_name, as_float=True, min_rows=0)
        return _result_stat(float(np.sum(arr)), spec)


@register_metric("unique_count")
class UniqueCountMetric:
    """Count of distinct values in a feature column."""

    name = "unique_count"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset(
        {ColumnType.numeric, ColumnType.categorical, ColumnType.binary}
    )

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the count of unique values.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a non-negative integer value.

        Raises:
            SchemaError: If ``spec.feature_name`` is ``None`` or absent from the DataFrame.
        """
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=0)
        return _result_stat(int(len(np.unique(arr))), spec)


@register_metric("in_range_count")
class InRangeCountMetric:
    """Count of values falling within a closed interval [low, high].

    Interval bounds are read from ``spec.params``:
    - ``"low"``: lower bound inclusive (default ``-inf``).
    - ``"high"``: upper bound inclusive (default ``+inf``).
    """

    name = "in_range_count"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the count of in-range values.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["low"]`` and ``params["high"]`` set
                inclusive range bounds (default ``-inf`` / ``+inf``).

        Returns:
            MetricResult with a non-negative integer value.

        Raises:
            SchemaError: If ``spec.feature_name`` is ``None`` or absent from the DataFrame.
        """
        low = float(spec.params.get("low", float("-inf")))
        high = float(spec.params.get("high", float("inf")))
        arr = extract_feature(current, spec.feature_name, as_float=True, min_rows=0)
        return _result_stat(int(np.sum((arr >= low) & (arr <= high))), spec)


@register_metric("out_range_count")
class OutRangeCountMetric:
    """Count of values falling outside a closed interval [low, high].

    Interval bounds are read from ``spec.params``:
    - ``"low"``: lower bound inclusive (default ``-inf``).
    - ``"high"``: upper bound inclusive (default ``+inf``).
    """

    name = "out_range_count"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset({ColumnType.numeric, ColumnType.binary})

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the count of out-of-range values.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["low"]`` and ``params["high"]`` set
                inclusive range bounds.

        Returns:
            MetricResult with a non-negative integer value.

        Raises:
            SchemaError: If ``spec.feature_name`` is ``None`` or absent from the DataFrame.
        """
        low = float(spec.params.get("low", float("-inf")))
        high = float(spec.params.get("high", float("inf")))
        arr = extract_feature(current, spec.feature_name, as_float=True, min_rows=0)
        return _result_stat(int(np.sum((arr < low) | (arr > high))), spec)


@register_metric("in_list_count")
class InListCountMetric:
    """Count of values matching any entry in a reference list.

    The list is read from ``spec.params["values"]`` (default empty list).
    Works on any column type.
    """

    name = "in_list_count"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset(
        {ColumnType.numeric, ColumnType.categorical, ColumnType.binary}
    )

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the count of values in the reference list.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema (must contain ``spec.feature_name``).
            spec: MetricSpec; ``params["values"]`` is the list of allowed
                values (default empty — returns 0).

        Returns:
            MetricResult with a non-negative integer value.

        Raises:
            SchemaError: If ``spec.feature_name`` is ``None`` or absent from the DataFrame.
        """
        values = spec.params.get("values", [])
        arr = extract_feature(current, spec.feature_name, as_float=False, min_rows=0)
        return _result_stat(int(np.sum(np.isin(arr, values))), spec)


@register_metric("row_count")
class RowCountMetric:
    """Total row count of the current window DataFrame.

    Dataset-level metric: ``spec.feature_name`` should be ``None``.
    """

    name = "row_count"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset()

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the total row count.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a non-negative integer value.
        """
        frame = nw.from_native(current, eager_only=True)
        return _result_stat(int(len(frame)), spec)


@register_metric("column_count")
class ColumnCountMetric:
    """Total column count of the current window DataFrame.

    Dataset-level metric: ``spec.feature_name`` should be ``None``.
    """

    name = "column_count"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset()

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the total column count.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a non-negative integer value.
        """
        frame = nw.from_native(current, eager_only=True)
        return _result_stat(int(len(frame.columns)), spec)


@register_metric("almost_constant_columns")
class AlmostConstantColumnsMetric:
    """Count of columns whose unique-value count is at or below a threshold.

    Dataset-level metric: ``spec.feature_name`` should be ``None``.
    ``spec.params["n_unique"]`` sets the threshold (default 1 — truly constant
    columns only).  Set to 2 to also flag near-constant columns.
    """

    name = "almost_constant_columns"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset()

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the count of almost-constant columns.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema.
            spec: MetricSpec; ``params["n_unique"]`` is the maximum number
                of distinct values allowed (default 1).

        Returns:
            MetricResult with a non-negative integer value.
        """
        frame = nw.from_native(current, eager_only=True)
        threshold = int(spec.params.get("n_unique", 1))
        unique_counts = frame.select(nw.all().n_unique()).row(0)
        count = sum(1 for n in unique_counts if n <= threshold)
        return _result_stat(int(count), spec)


@register_metric("duplicate_rows")
class DuplicateRowsMetric:
    """Count of duplicate rows in the current window DataFrame.

    Dataset-level metric: ``spec.feature_name`` should be ``None``.
    Count = total rows − unique rows; a row appearing 3 times contributes 2.
    """

    name = "duplicate_rows"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset()

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the count of duplicate rows.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a non-negative integer value.
        """
        frame = nw.from_native(current, eager_only=True)
        duplicate_count = len(frame) - len(frame.unique())
        return _result_stat(int(max(0, duplicate_count)), spec)


@register_metric("empty_columns")
class EmptyColumnsMetric:
    """Count of columns containing only null or NaN values.

    Dataset-level metric: ``spec.feature_name`` should be ``None``.
    A column is empty when its null count equals the total row count.
    """

    name = "empty_columns"
    metric_type = MetricType.statistics
    requires_reference = False
    accepted_column_types: frozenset[ColumnType] = frozenset()

    def compute(self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec) -> MetricResult:
        """Compute the count of empty columns.

        Args:
            current: Current-window DataFrame.
            reference: Ignored.
            schema: DataSchema.
            spec: MetricSpec; supports optional threshold.

        Returns:
            MetricResult with a non-negative integer value.
        """
        frame = nw.from_native(current, eager_only=True)
        n_rows = len(frame)
        null_counts = frame.select(nw.all().null_count()).row(0)
        count = sum(1 for n in null_counts if n == n_rows)
        return _result_stat(int(count), spec)
