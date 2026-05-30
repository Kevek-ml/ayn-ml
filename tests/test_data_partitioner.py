from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ayn_ml.core.schema import TabularSchema
from ayn_ml.data.partitioner import FixedReferencePartitioner, TimeBasedPartitioner
from ayn_ml.exceptions import InsufficientDataError, SchemaError


@pytest.fixture
def schema():
    return TabularSchema(timestamp_col="timestamp")


@pytest.fixture
def current_df():
    return pd.DataFrame({"value": [1, 2, 3]})


@pytest.fixture
def reference_df():
    return pd.DataFrame({"value": [4, 5, 6]})


@pytest.fixture
def ts_df():
    dates = pd.date_range("2024-01-01", periods=20, freq="D", tz=timezone.utc)
    return pd.DataFrame({"timestamp": dates, "value": range(20)})


class TestFixedReferencePartitioner:
    def test_returns_correct_pair(self, current_df, reference_df, schema):
        current, ref = FixedReferencePartitioner(reference_df).partition(current_df, schema)
        assert len(current) == 3
        assert len(ref) == 3

    def test_current_is_input_df(self, current_df, reference_df, schema):
        current, _ = FixedReferencePartitioner(reference_df).partition(current_df, schema)
        assert current is current_df

    def test_reference_is_stored_df(self, current_df, reference_df, schema):
        _, ref = FixedReferencePartitioner(reference_df).partition(current_df, schema)
        assert ref is reference_df


class TestTimeBasedPartitioner:
    def test_splits_correctly(self, ts_df, schema):
        cutoff = datetime(2024, 1, 11, tzinfo=timezone.utc)
        window = timedelta(days=5)
        current, ref = TimeBasedPartitioner(cutoff, window).partition(ts_df, schema)
        assert len(current) == 9  # Jan 12–20
        assert len(ref) == 5  # Jan 6–10

    def test_reference_none_when_no_ref_rows(self, ts_df, schema):
        # cutoff before all rows → current = all rows, reference window contains none
        cutoff = datetime(2023, 12, 31, tzinfo=timezone.utc)
        window = timedelta(days=1)
        _, ref = TimeBasedPartitioner(cutoff, window).partition(ts_df, schema)
        assert ref is None

    def test_raises_when_no_current_rows(self, ts_df, schema):
        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(InsufficientDataError):
            TimeBasedPartitioner(cutoff, timedelta(days=30)).partition(ts_df, schema)

    def test_raises_when_timestamp_col_missing(self, current_df, schema):
        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(SchemaError, match="timestamp"):
            TimeBasedPartitioner(cutoff, timedelta(days=10)).partition(current_df, schema)

    def test_raises_on_non_positive_window(self):
        with pytest.raises(ValueError):
            TimeBasedPartitioner(datetime(2024, 1, 1, tzinfo=timezone.utc), timedelta(seconds=0))

    def test_row_at_cutoff_goes_to_reference(self, ts_df, schema):
        # Jan 11 is in ts_df; rows at exactly cutoff belong to reference, not current
        cutoff = datetime(2024, 1, 11, tzinfo=timezone.utc)
        current, ref = TimeBasedPartitioner(cutoff, timedelta(days=5)).partition(ts_df, schema)
        # current starts Jan 12 (strict >), reference ends Jan 11 (inclusive <=)
        assert current["timestamp"].min() > pd.Timestamp(cutoff)
        assert ref["timestamp"].max() == pd.Timestamp(cutoff)
