"""CSV file data source for ayn-ml.

``CsvSource`` reads a CSV file from disk and returns a narwhals-compatible
DataFrame projected to only the columns required by the monitoring plan.

Backend selection
-----------------
Set ``backend`` to any narwhals-supported eager backend name (``"polars"``,
``"pandas"``, ``"modin"``, ``"cudf"``, ``"pyarrow"``).  The default
``"auto"`` tries Polars first and falls back to pandas.

``separator`` is a first-class parameter handled uniformly by narwhals across
all backends.  All other backend-specific options go in ``read_kwargs`` and
are forwarded verbatim to the native reader — the user is responsible for
using kwargs that match their chosen backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import narwhals as nw

from ayn_ml.data.source import DataSource, required_columns

if TYPE_CHECKING:
    from ayn_ml.core.spec import MonitoringPlan

logger = logging.getLogger(__name__)

# Preference order for backend="auto".
_AUTO_PREFERENCE: tuple[str, ...] = ("polars", "pandas")


@dataclass(frozen=True)
class CsvSource(DataSource):
    """Data source that reads a CSV file from disk.

    Backend selection is explicit: set ``backend`` to any narwhals-supported
    eager backend (``"polars"``, ``"pandas"``, ``"modin"``, ``"cudf"``,
    ``"pyarrow"``).  The default ``"auto"`` tries Polars first and falls back
    to pandas.

    ``separator`` is translated correctly for every backend by narwhals.
    Additional options in ``read_kwargs`` are forwarded verbatim to the
    native reader — match them to your chosen backend's API.

    Args:
        path: Path to the CSV file.
        backend: Backend to use for reading.  One of ``"auto"``, ``"polars"``,
            ``"pandas"``, ``"modin"``, ``"cudf"``, ``"pyarrow"``, or any
            other narwhals-supported eager backend name.  Defaults to
            ``"auto"`` (Polars if available, else pandas).
        separator: Single character column separator.  Translated to the
            correct kwarg name (``separator`` / ``sep``) by narwhals.
            Defaults to ``","``.
        read_kwargs: Extra keyword arguments forwarded verbatim to the
            native backend reader.  Use kwargs that match ``backend``'s API.
            When ``backend="auto"``, pass kwargs compatible with whichever
            backend will be selected at runtime (a warning is emitted to
            help you identify the active backend).
    """

    path: str | Path
    backend: str = "auto"
    separator: str = ","
    read_kwargs: dict[str, Any] = field(default_factory=dict)

    def load(self, plan: MonitoringPlan) -> Any:
        """Read the CSV file and return a projected narwhals-compatible frame.

        Reads the CSV via narwhals (which handles backend dispatch and
        ``separator`` normalisation), then projects to the columns required
        by ``plan``.  Columns listed by ``required_columns`` that are absent
        from the file are silently skipped.

        Args:
            plan: MonitoringPlan providing schema and metric definitions used
                to determine which columns to project.

        Returns:
            A native DataFrame (Polars, pandas, or whichever backend was
            selected) containing at least the columns required by the plan
            that are present in the CSV file.

        Raises:
            FileNotFoundError: If ``path`` does not point to an existing file.
            ImportError: If a specific backend is requested but not installed.
            Exception: Parse errors are re-raised as-is from the native
                backend reader.
        """
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        if self.backend != "auto":
            logger.debug("CsvSource: reading %s (backend=%r)", path, self.backend)
            nw_df = nw.read_csv(
                path,
                backend=self.backend,
                separator=self.separator,
                **self.read_kwargs,
            )
        else:
            last_err: Exception | None = None
            for backend in _AUTO_PREFERENCE:
                try:
                    nw_df = nw.read_csv(
                        path,
                        backend=backend,
                        separator=self.separator,
                        **self.read_kwargs,
                    )
                except ImportError as exc:
                    last_err = exc
                    continue
                if self.read_kwargs:
                    logger.warning(
                        "CsvSource: backend='auto' selected %r with read_kwargs keys=%s. "
                        "Set backend=%r explicitly to avoid surprises if your environment changes.",
                        backend,
                        list(self.read_kwargs.keys()),
                        backend,
                    )
                else:
                    logger.debug("CsvSource: auto selected %r backend", backend)
                break
            else:
                raise ImportError(
                    "CsvSource: no backend available. "
                    "Install polars (pip install polars) or pandas (pip install pandas)."
                ) from last_err

        cols = required_columns(plan)
        if not cols:
            logger.warning(
                "CsvSource: required_columns returned an empty list for plan %r — "
                "check that data_schema and metrics are configured correctly",
                plan.name,
            )
        present = [c for c in cols if c in nw_df.columns]
        logger.debug(
            "CsvSource: projecting %d/%d required columns from %s",
            len(present),
            len(cols),
            path,
        )
        return nw.to_native(nw_df.select(present))
