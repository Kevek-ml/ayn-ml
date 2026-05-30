"""Data loading, sampling, and partitioning for ayn-ml monitoring runs.

Public API
----------
DataSource
    Abstract base; implement to add custom backends (databases, feature stores).
DataFrameSource
    Wraps an already-loaded in-memory DataFrame.
CsvSource
    Reads a CSV file from disk; tries Polars first, falls back to pandas.
ExcelSource
    Reads a worksheet from an Excel file (opt-in: ``pip install ayn-ml[excel]``).
required_columns
    Computes the minimal column projection for a schema + plan pair.

SamplingStrategy
    Abstract base for current-window extraction.
FullDataSampling
    Identity — returns all rows.
LastNRowsSampling
    Returns the last N rows by position.
TimeWindowSampling
    Filters on a timestamp range.
RandomSampling
    Randomly subsamples rows for performance.

DataPartitioner
    Abstract base for current/reference splitting.
FixedReferencePartitioner
    Separate reference DataFrame supplied at construction time.
TimeBasedPartitioner
    Splits a single DataFrame on a timestamp cutoff.
"""

from ayn_ml.data.csv import CsvSource
from ayn_ml.data.excel import ExcelSource
from ayn_ml.data.partitioner import (
    DataPartitioner,
    FixedReferencePartitioner,
    TimeBasedPartitioner,
)
from ayn_ml.data.sampling import (
    FullDataSampling,
    LastNRowsSampling,
    RandomSampling,
    SamplingStrategy,
    TimeWindowSampling,
)
from ayn_ml.data.source import DataFrameSource, DataSource, required_columns

__all__ = [
    "DataSource",
    "DataFrameSource",
    "CsvSource",
    "ExcelSource",
    "required_columns",
    "SamplingStrategy",
    "FullDataSampling",
    "LastNRowsSampling",
    "RandomSampling",
    "TimeWindowSampling",
    "DataPartitioner",
    "FixedReferencePartitioner",
    "TimeBasedPartitioner",
]
