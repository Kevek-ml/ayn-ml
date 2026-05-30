from datetime import datetime, timezone

import pandas as pd
import pytest

from ayn_ml.core.schema import TabularSchema
from ayn_ml.data.sampling import FullDataSampling, LastNRowsSampling, RandomSampling, TimeWindowSampling
from ayn_ml.exceptions import InsufficientDataError, SchemaError


@pytest.fixture
def schema():
    return TabularSchema(timestamp_col="timestamp")


@pytest.fixture
def df():
    return pd.DataFrame({"a": range(10)})


@pytest.fixture
def ts_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz=timezone.utc)
    return pd.DataFrame({"timestamp": dates, "value": range(10)})


class TestFullDataSampling:
    def test_returns_all_rows(self, df, schema):
        result = FullDataSampling().sample(df, schema)
        assert len(result) == len(df)

    def test_returns_same_object(self, df, schema):
        result = FullDataSampling().sample(df, schema)
        assert result is df


class TestLastNRowsSampling:
    def test_returns_last_n_rows(self, df, schema):
        result = LastNRowsSampling(3).sample(df, schema)
        assert len(result) == 3
        assert list(result["a"]) == [7, 8, 9]

    def test_raises_when_insufficient(self, df, schema):
        with pytest.raises(InsufficientDataError):
            LastNRowsSampling(100).sample(df, schema)

    def test_raises_on_non_positive_n(self):
        with pytest.raises(ValueError):
            LastNRowsSampling(0)

    def test_exact_n_rows(self, df, schema):
        result = LastNRowsSampling(10).sample(df, schema)
        assert len(result) == 10


class TestTimeWindowSampling:
    def test_filters_to_window(self, ts_df, schema):
        start = datetime(2024, 1, 3, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        result = TimeWindowSampling(start, end).sample(ts_df, schema)
        assert len(result) == 3

    def test_raises_when_window_empty(self, ts_df, schema):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        with pytest.raises(InsufficientDataError):
            TimeWindowSampling(start, end).sample(ts_df, schema)

    def test_raises_when_timestamp_col_missing(self, df, schema):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, tzinfo=timezone.utc)
        with pytest.raises(SchemaError, match="timestamp"):
            TimeWindowSampling(start, end).sample(df, schema)

    def test_raises_when_start_not_before_end(self):
        t = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            TimeWindowSampling(t, t)


class TestRandomSampling:
    def test_sample_n_returns_correct_count(self, df, schema):
        result = RandomSampling(n=5).sample(df, schema)
        assert len(result) == 5

    def test_sample_n_larger_than_df_clips_to_df_size(self, df, schema):
        result = RandomSampling(n=100).sample(df, schema)
        assert len(result) == len(df)

    def test_sample_frac(self, df, schema):
        result = RandomSampling(frac=0.5).sample(df, schema)
        assert len(result) == 5

    def test_sample_is_reproducible_with_seed(self, df, schema):
        r1 = RandomSampling(n=5, seed=42).sample(df, schema)
        r2 = RandomSampling(n=5, seed=42).sample(df, schema)
        assert list(r1["a"]) == list(r2["a"])

    def test_raises_when_both_n_and_frac_set(self):
        with pytest.raises(ValueError):
            RandomSampling(n=5, frac=0.5)

    def test_raises_when_neither_set(self):
        with pytest.raises(ValueError):
            RandomSampling()

    def test_raises_on_non_positive_n(self):
        with pytest.raises(ValueError):
            RandomSampling(n=0)

    def test_raises_on_invalid_frac(self):
        with pytest.raises(ValueError):
            RandomSampling(frac=1.5)

    def test_raises_on_empty_df(self, schema):
        import pandas as pd

        empty = pd.DataFrame({"a": []})
        with pytest.raises(InsufficientDataError):
            RandomSampling(n=5).sample(empty, schema)
