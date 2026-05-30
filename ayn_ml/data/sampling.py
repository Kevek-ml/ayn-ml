"""Sampling strategies for extracting a current window from a DataFrame.

A ``SamplingStrategy`` narrows a full DataFrame to the rows that represent
the current monitoring window.  Four strategies are provided:

- ``FullDataSampling``   — identity, returns all rows.
- ``LastNRowsSampling``  — returns the last N rows (by position, not sorted).
- ``TimeWindowSampling`` — filters on a timestamp column.
- ``RandomSampling``     — randomly subsamples rows for performance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import narwhals as nw

from ayn_ml.core.schema import DataSchema
from ayn_ml.exceptions import InsufficientDataError, SchemaError


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies.

    All strategies accept a DataFrame and a DataSchema and return a
    (potentially smaller) DataFrame representing the current window.
    """

    @abstractmethod
    def sample(self, df: Any, schema: DataSchema) -> Any:
        """Extract the current window from a DataFrame.

        Args:
            df: Source DataFrame (pandas or Polars).
            schema: DataSchema providing column name mappings.

        Returns:
            A narwhals-compatible DataFrame containing the sampled rows.

        Raises:
            InsufficientDataError: If the resulting window is empty or too
                small to be meaningful.
            SchemaError: If a required column (e.g. timestamp) is absent.
        """


class FullDataSampling(SamplingStrategy):
    """Returns all rows unchanged — the identity sampling strategy.

    Use this when the DataFrame already represents exactly the window you
    want to monitor (the common case when data is pre-filtered upstream).
    """

    def sample(self, df: Any, schema: DataSchema) -> Any:
        """Return the DataFrame unchanged.

        Args:
            df: Source DataFrame.
            schema: DataSchema (unused).

        Returns:
            The original DataFrame.
        """
        return df


class LastNRowsSampling(SamplingStrategy):
    """Returns the last N rows by position (no sorting applied).

    Assumes the DataFrame is already in chronological order.  Use
    ``TimeWindowSampling`` when an explicit timestamp range is preferred.

    Args:
        n: Number of rows to keep from the tail of the DataFrame.
    """

    def __init__(self, n: int) -> None:
        """Initialise with the desired window size.

        Args:
            n: Number of tail rows to return.  Must be positive.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}.")
        self.n = n

    def sample(self, df: Any, schema: DataSchema) -> Any:
        """Return the last ``n`` rows.

        Args:
            df: Source DataFrame.
            schema: DataSchema (unused).

        Returns:
            DataFrame containing the last ``n`` rows.

        Raises:
            InsufficientDataError: If the DataFrame has fewer than ``n`` rows.
        """
        native = nw.from_native(df, eager_only=True)
        if len(native) < self.n:
            raise InsufficientDataError(f"DataFrame has {len(native)} rows; need at least {self.n}.")
        return nw.to_native(native.tail(self.n))


class TimeWindowSampling(SamplingStrategy):
    """Returns rows where the timestamp column falls within [start, end].

    Args:
        start: Window start (inclusive).
        end: Window end (inclusive).
    """

    def __init__(self, start: datetime, end: datetime) -> None:
        """Initialise with a time range.

        Args:
            start: Window start (inclusive).
            end: Window end (inclusive).

        Raises:
            ValueError: If ``start`` is not strictly before ``end``.
        """
        if start >= end:
            raise ValueError(f"start must be before end, got {start} >= {end}.")
        self.start = start
        self.end = end

    def sample(self, df: Any, schema: DataSchema) -> Any:
        """Filter rows to those within [start, end] on the timestamp column.

        Args:
            df: Source DataFrame.
            schema: DataSchema; ``timestamp_col`` must be present.

        Returns:
            Filtered DataFrame.

        Raises:
            SchemaError: If ``schema.timestamp_col`` is absent from the frame.
            InsufficientDataError: If no rows fall within the window.
        """
        ts_col = schema.timestamp_col
        native = nw.from_native(df, eager_only=True)
        if ts_col not in native.columns:
            raise SchemaError(f"Timestamp column '{ts_col}' not found in DataFrame.")
        filtered = native.filter((nw.col(ts_col) >= self.start) & (nw.col(ts_col) <= self.end))
        if len(filtered) == 0:
            raise InsufficientDataError(f"No rows found in time window [{self.start}, {self.end}].")
        return nw.to_native(filtered)


class RandomSampling(SamplingStrategy):
    """Randomly subsamples rows for performance.

    Applied after window selection to reduce the current window to a
    manageable size before metrics are computed.  Exactly one of ``n`` or
    ``frac`` must be provided.

    Args:
        n: Absolute number of rows to sample.  Must be positive.
        frac: Fraction of rows to sample.  Must be in ``(0, 1]``.
        seed: Random seed for reproducibility.  ``None`` means non-deterministic.
    """

    def __init__(self, n: int | None = None, frac: float | None = None, seed: int | None = None) -> None:
        """Initialise with a sample size or fraction.

        Args:
            n: Absolute number of rows to keep.  Must be positive.
            frac: Fraction of rows to keep.  Must be in ``(0, 1]``.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If neither or both of ``n`` and ``frac`` are set, or
                if their values are out of range.
        """
        if (n is None) == (frac is None):
            raise ValueError("Exactly one of 'n' or 'frac' must be set.")
        if n is not None and n <= 0:
            raise ValueError(f"n must be positive, got {n}.")
        if frac is not None and not (0 < frac <= 1):
            raise ValueError(f"frac must be in (0, 1], got {frac}.")
        self.n = n
        self.frac = frac
        self.seed = seed

    def sample(self, df: Any, schema: DataSchema) -> Any:
        """Return a random subset of rows.

        Args:
            df: Source DataFrame.
            schema: DataSchema (unused).

        Returns:
            DataFrame containing the sampled rows.

        Raises:
            InsufficientDataError: If the DataFrame is empty or the computed
                sample size is zero.
        """
        native = nw.from_native(df, eager_only=True)
        if len(native) == 0:
            raise InsufficientDataError("Cannot sample from an empty DataFrame.")
        if self.n is not None:
            size = min(self.n, len(native))
        else:
            size = max(1, int(len(native) * self.frac))  # type: ignore[arg-type]
        return nw.to_native(native.sample(n=size, seed=self.seed))
