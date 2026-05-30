"""Data partitioners for splitting a DataFrame into current and reference windows.

A ``DataPartitioner`` takes a single DataFrame and splits it into two windows:

- **current** — the window to evaluate (recent production data).
- **reference** — the baseline to compare against (training distribution or
  a prior production window).  ``None`` when the plan contains only
  performance metrics that do not require a reference.

Two partitioners are provided:

- ``FixedReferencePartitioner`` — current data and reference data are supplied
  as separate DataFrames (the most common case).
- ``TimeBasedPartitioner`` — splits a single DataFrame on a timestamp cutoff.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import narwhals as nw

from ayn_ml.core.schema import DataSchema
from ayn_ml.exceptions import InsufficientDataError, SchemaError


class DataPartitioner(ABC):
    """Abstract base class for data partitioners.

    Returns a ``(current, reference)`` tuple.  ``reference`` is ``None``
    when the source data does not support a reference window (e.g. a
    performance-only plan with no drift metrics).
    """

    @abstractmethod
    def partition(self, df: Any, schema: DataSchema) -> tuple[Any, Any | None]:
        """Split ``df`` into current and reference windows.

        Args:
            df: Source DataFrame (pandas or Polars).
            schema: DataSchema providing column name mappings.

        Returns:
            A ``(current, reference)`` tuple.  ``reference`` is ``None`` when
            no reference window can be derived.

        Raises:
            SchemaError: If a required column (e.g. timestamp) is absent.
            InsufficientDataError: If the current window is empty.
        """


class FixedReferencePartitioner(DataPartitioner):
    """Uses a pre-loaded DataFrame as the fixed reference window.

    This is the standard choice when the reference distribution is known
    upfront (e.g. training data or a frozen baseline snapshot).

    Args:
        reference: Reference DataFrame (pandas or Polars).
    """

    def __init__(self, reference: Any) -> None:
        """Initialise with a fixed reference DataFrame.

        Args:
            reference: Reference DataFrame.
        """
        self._reference = reference

    def partition(self, df: Any, schema: DataSchema) -> tuple[Any, Any | None]:
        """Return ``(df, reference)`` unchanged.

        Args:
            df: Current-window DataFrame.
            schema: DataSchema (unused).

        Returns:
            ``(df, self._reference)`` tuple.
        """
        return df, self._reference


class TimeBasedPartitioner(DataPartitioner):
    """Splits a single DataFrame on a timestamp cutoff.

    Rows with timestamp **after** ``cutoff`` form the current window.
    Rows with timestamp in ``(cutoff - reference_window, cutoff]`` form the
    reference window.  Both windows are derived from the same source frame.

    Args:
        cutoff: Boundary between reference and current data.
        reference_window: Duration of the reference window before the cutoff.
    """

    def __init__(self, cutoff: datetime, reference_window: timedelta) -> None:
        """Initialise with a cutoff and a reference window duration.

        Args:
            cutoff: Rows after this timestamp are treated as current.
            reference_window: Duration of the reference period ending at
                ``cutoff``.  Must be positive.

        Raises:
            ValueError: If ``reference_window`` is not positive.
        """
        if reference_window.total_seconds() <= 0:
            raise ValueError("reference_window must be positive.")
        self.cutoff = cutoff
        self.reference_window = reference_window

    def partition(self, df: Any, schema: DataSchema) -> tuple[Any, Any | None]:
        """Split ``df`` into current (post-cutoff) and reference (pre-cutoff) windows.

        Args:
            df: Source DataFrame containing both windows.
            schema: DataSchema; ``timestamp_col`` must be present.

        Returns:
            ``(current, reference)`` tuple.  ``reference`` is ``None`` when
            no rows fall within the reference period.  Rows timestamped
            exactly at ``cutoff`` are assigned to the reference window
            (``ts <= cutoff``); the current window uses a strict greater-than
            (``ts > cutoff``) so no row is double-counted.

        Raises:
            SchemaError: If ``schema.timestamp_col`` is absent from the frame.
            InsufficientDataError: If no rows fall after the cutoff.
        """
        ts_col = schema.timestamp_col
        native = nw.from_native(df, eager_only=True)
        if ts_col not in native.columns:
            raise SchemaError(f"Timestamp column '{ts_col}' not found; required for time-based partitioning.")

        current_native = native.filter(nw.col(ts_col) > self.cutoff)
        if len(current_native) == 0:
            raise InsufficientDataError(f"No rows found after cutoff {self.cutoff}; cannot form a current window.")

        ref_start = self.cutoff - self.reference_window
        ref_native = native.filter((nw.col(ts_col) > ref_start) & (nw.col(ts_col) <= self.cutoff))

        reference = nw.to_native(ref_native) if len(ref_native) > 0 else None
        return nw.to_native(current_native), reference
