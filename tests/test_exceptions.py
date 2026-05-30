import pytest

from ayn_ml.exceptions import (
    AynError,
    InsufficientDataError,
    MetricComputeError,
    SchemaError,
    ThresholdError,
    UnknownMetricError,
)


def test_all_exceptions_are_ayn_error():
    for cls in (
        UnknownMetricError,
        SchemaError,
        ThresholdError,
        InsufficientDataError,
        MetricComputeError,
    ):
        assert issubclass(cls, AynError)
        assert issubclass(cls, Exception)


def test_exceptions_carry_message():
    err = SchemaError("column 'y_true' not found")
    assert "y_true" in str(err)


def test_exceptions_are_catchable_as_ayn_error():
    with pytest.raises(AynError):
        raise UnknownMetricError("accuracy")

    with pytest.raises(AynError):
        raise MetricComputeError("divide by zero")
