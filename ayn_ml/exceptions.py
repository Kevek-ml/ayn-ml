"""Custom exception hierarchy for ayn-ml.

All exceptions inherit from AynError so callers can catch the full library
surface with a single except clause when desired.
"""


class AynError(Exception):
    """Base exception for all ayn-ml errors.

    Catch this class to handle any library-specific error uniformly.
    """


class UnknownMetricError(AynError):
    """Raised when a metric name is not found in the registry.

    Attributes:
        message: Human-readable description including the list of known metrics.
    """


class SchemaError(AynError):
    """Raised when a required column is missing or the schema is incompatible.

    Examples:
        Raised when TabularSchema.label_col does not exist in the DataFrame,
        or when a drift metric is given a TextSchema.
    """


class ThresholdError(AynError):
    """Raised when a MetricSpec contains an invalid threshold configuration.

    Examples:
        Raised when a list threshold has fewer than two elements.
    """


class InsufficientDataError(AynError):
    """Raised when the DataFrame has too few rows to compute a metric reliably.

    Examples:
        Raised when fewer than 2 rows are passed to a performance metric,
        or when all y_true values are zero for MAPE.
    """


class MetricComputeError(AynError):
    """Raised inside compute() for unexpected errors not covered by other types.

    The Runner catches this per-metric and records it as a MetricError rather
    than aborting the full monitoring run.
    """
