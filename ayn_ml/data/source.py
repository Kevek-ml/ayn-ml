"""Data source abstractions for ayn-ml.

A ``DataSource`` knows how to produce a DataFrame for a monitoring run.
``DataFrameSource`` wraps an already-loaded in-memory DataFrame — the simplest
case and the entry point for users who load data themselves.

Column projection
-----------------
``DataFrameSource.load`` receives the full ``MonitoringPlan`` and calls
``required_columns`` internally to project only the columns needed: mandatory
schema columns and any ``feature_name`` values declared in metric specs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    from ayn_ml.core.spec import MonitoringPlan


class DataSource(ABC):
    """Abstract base class for data sources.

    Subclasses implement ``load`` to fetch a DataFrame from any backend
    (in-memory, Parquet, database, feature store, etc.).
    """

    @abstractmethod
    def load(self, plan: MonitoringPlan) -> Any:
        """Load a DataFrame projected to the columns required by ``plan``.

        Args:
            plan: MonitoringPlan that owns this run.  The source uses it to
                determine which columns to load (via ``required_columns``).

        Returns:
            A narwhals-compatible DataFrame (pandas or Polars) containing at
            least the columns required by the plan.  Columns absent from the
            source are silently skipped.
        """


class DataFrameSource(DataSource):
    """Wraps an already-loaded in-memory DataFrame.

    This is the primary entry point for users who load data themselves before
    passing it to the runner.  ``load`` projects the DataFrame to only the
    columns required by the plan, avoiding unnecessary copies on wide tables.

    Args:
        df: Source DataFrame (pandas or Polars).
    """

    def __init__(self, df: Any) -> None:
        """Initialise with an in-memory DataFrame.

        Args:
            df: Source DataFrame (pandas or Polars eager frame).
        """
        self._df = df

    def load(self, plan: MonitoringPlan) -> Any:
        """Return the wrapped DataFrame projected to the columns needed by ``plan``.

        Args:
            plan: MonitoringPlan providing schema and metrics.

        Returns:
            The source DataFrame restricted to the required columns.  Columns
            listed by ``required_columns`` that are absent from the source are
            silently skipped.
        """
        cols = required_columns(plan)
        native = nw.from_native(self._df, eager_only=True)
        present = [c for c in cols if c in native.columns]
        return nw.to_native(native.select(present))


def required_columns(plan: MonitoringPlan) -> list[str]:
    """Compute the minimal ordered column set needed for a monitoring run.

    Collects columns from three sources, in order, deduplicating while
    preserving insertion order:

    1. All non-``None`` column names declared by ``plan.data_schema``
       (via ``schema.column_names``).
    2. Any ``feature_name`` values declared on individual ``MetricSpec``
       entries in ``plan.metrics``.
    3. Any column names listed in ``spec.params["item_features"]`` — used
       by recsys metrics such as ``diversity`` and ``serendipity`` that
       require item feature columns to be present in the DataFrame.

    Args:
        plan: MonitoringPlan describing the schema and metrics for the run.

    Returns:
        Deduplicated list of column names in a stable order.  The runner
        passes this to ``DataSource.load`` so projection happens at source
        time rather than after a full table scan.
    """
    seen: dict[str, None] = {}

    for col in plan.data_schema.column_names:
        seen[col] = None

    for spec in plan.metrics:
        if spec.feature_name:
            seen[spec.feature_name] = None
        for col in spec.params.get("item_features", []):
            if isinstance(col, str):
                seen[col] = None

    return list(seen)
