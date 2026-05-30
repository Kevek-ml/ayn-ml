"""Window selection, random sampling, and partitioning configuration models.

These Pydantic models live inside ``MonitoringPlan`` and declare *how* the
runner should narrow and split its data before computing metrics.  Being
plain Pydantic models they round-trip through JSON / YAML unchanged and
carry no runtime dependencies on the data layer.

The actual strategy implementations are in ``ayn_ml.data``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FullWindowConfig(BaseModel):
    """Pass the full DataFrame through unchanged.

    Use when the DataFrame is already pre-filtered to the desired window.
    """

    type: Literal["full"] = "full"


class LastNRowsWindowConfig(BaseModel):
    """Keep the last N rows by position (no sorting applied).

    Assumes the DataFrame is already in chronological order.

    Attributes:
        n: Number of tail rows to keep.  Must be positive.
    """

    type: Literal["last_n"] = "last_n"
    n: int = Field(gt=0)


class TimeWindowConfig(BaseModel):
    """Keep rows where the timestamp column falls in ``[start, end]``.

    Attributes:
        start: Window start (inclusive).
        end: Window end (inclusive).
    """

    type: Literal["time_window"] = "time_window"
    start: datetime
    end: datetime

    @field_validator("end")
    @classmethod
    def end_after_start(cls, end: datetime, info: object) -> datetime:
        """Validate that end is strictly after start."""
        start = getattr(info, "data", {}).get("start")
        if start is not None and end <= start:
            raise ValueError(f"end must be after start, got {start} >= {end}.")
        return end


WindowConfig = Annotated[
    FullWindowConfig | LastNRowsWindowConfig | TimeWindowConfig,
    Field(discriminator="type"),
]
"""Discriminated union of all window configuration types.

Use this annotation on ``MonitoringPlan.window``.  Pydantic selects the
correct subclass via the ``type`` field when deserializing from JSON or YAML.
"""


class RandomSamplingConfig(BaseModel):
    """Randomly subsample rows for performance.

    Reduces the current window to a manageable size before metrics are
    computed.  Exactly one of ``n`` or ``frac`` must be provided.

    Attributes:
        n: Absolute number of rows to sample.  Must be positive.
        frac: Fraction of rows to sample.  Must be in ``(0, 1]``.
        seed: Random seed for reproducibility.  ``None`` means non-deterministic.
    """

    type: Literal["random"] = "random"
    n: int | None = None
    frac: float | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def exactly_one_of_n_or_frac(self) -> RandomSamplingConfig:
        """Validate that exactly one of n or frac is set."""
        if (self.n is None) == (self.frac is None):
            raise ValueError("Exactly one of 'n' or 'frac' must be set.")
        if self.n is not None and self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}.")
        if self.frac is not None and not (0 < self.frac <= 1):
            raise ValueError(f"frac must be in (0, 1], got {self.frac}.")
        return self


class TimeBasedPartitioningConfig(BaseModel):
    """Split a single DataFrame on a timestamp cutoff.

    Rows strictly after ``cutoff`` form the current window.  Rows in
    ``(cutoff - reference_window, cutoff]`` form the reference window.
    Rows at exactly ``cutoff`` are assigned to the reference window so no
    row is counted twice.

    Attributes:
        cutoff: Boundary between reference and current data.
        reference_window: Duration of the reference period ending at
            ``cutoff``.  Must be positive.
    """

    type: Literal["time_based"] = "time_based"
    cutoff: datetime
    reference_window: timedelta


PartitioningConfig = TimeBasedPartitioningConfig
"""Type alias kept for backwards-compatibility.

DataPartitioner implementations (e.g. TimeBasedPartitioner) are instantiated
directly at runtime — partitioning is not declared on MonitoringPlan.
"""
