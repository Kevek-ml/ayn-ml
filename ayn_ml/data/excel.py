"""Excel file data source for ayn-ml.

``ExcelSource`` reads a worksheet from an Excel file and returns a
narwhals-compatible DataFrame projected to the columns required by the
monitoring plan.

Backend selection
-----------------
Set ``backend`` to ``"polars"`` or ``"pandas"``.  The default ``"auto"``
tries Polars first (via ``fastexcel``) and falls back to pandas (via
``openpyxl``).

Opt-in dependencies
-------------------
* Polars backend: ``pip install ayn-ml[excel]`` (installs ``fastexcel``)
* pandas backend: ``pip install ayn-ml[excel]`` (installs ``openpyxl``)

Both are included in the ``[excel]`` extra.
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
class ExcelSource(DataSource):
    """Load a worksheet from an Excel file.

    Supports ``.xlsx``, ``.xls``, and other formats understood by the active
    backend.  Backend selection is explicit: set ``backend`` to ``"polars"``
    (uses ``fastexcel`` / calamine — fast, no Java) or ``"pandas"`` (uses
    ``openpyxl``).  The default ``"auto"`` tries Polars first and falls back
    to pandas.

    ``sheet_name`` uses pandas conventions: a string selects by name, an
    integer selects by 0-based position.  Both conventions are translated
    automatically when the Polars backend is active (Polars uses a 1-based
    ``sheet_id`` for integer selection).

    ``read_kwargs`` are forwarded verbatim to the native reader — use kwargs
    that match your chosen backend's API.

    Args:
        path: Path to the Excel file.
        backend: Backend to use.  ``"polars"`` (requires ``fastexcel``),
            ``"pandas"`` (requires ``openpyxl``), or ``"auto"`` (tries
            Polars first, falls back to pandas).  Defaults to ``"auto"``.
        sheet_name: Sheet to read — a name (``str``) or 0-based integer
            index.  Defaults to the first sheet (``0``).
        read_kwargs: Extra keyword arguments forwarded verbatim to the
            native reader (``pl.read_excel`` or ``pd.read_excel``).

    Example::

        source = ExcelSource(path="data/predictions.xlsx", sheet_name="run_42")
        df = source.load(plan)
    """

    path: str | Path
    backend: str = "auto"
    sheet_name: str | int = 0
    read_kwargs: dict[str, Any] = field(default_factory=dict)

    def load(self, plan: MonitoringPlan) -> Any:
        """Read the Excel worksheet and project to plan-required columns.

        Args:
            plan: MonitoringPlan providing schema and metrics used to
                determine which columns to load.

        Returns:
            A native DataFrame (Polars or pandas, depending on the active
            backend) restricted to the columns required by ``plan``.
            Columns present in the plan but absent from the worksheet are
            silently skipped.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ImportError: If the required backend dependency is not installed.
            Exception: Parse errors are re-raised as-is from the native reader.
        """
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        if self.backend != "auto":
            logger.debug("ExcelSource: reading %s (backend=%r)", path, self.backend)
            native_df = self._read_backend(self.backend, path)
        else:
            last_err: Exception | None = None
            for backend in _AUTO_PREFERENCE:
                try:
                    native_df = self._read_backend(backend, path)
                except ImportError as exc:
                    last_err = exc
                    continue
                if self.read_kwargs:
                    logger.warning(
                        "ExcelSource: backend='auto' selected %r with read_kwargs keys=%s. "
                        "Set backend=%r explicitly to avoid surprises if your environment changes.",
                        backend,
                        list(self.read_kwargs.keys()),
                        backend,
                    )
                else:
                    logger.debug("ExcelSource: auto selected %r backend", backend)
                break
            else:
                raise ImportError(
                    "ExcelSource: no backend available. "
                    "Install dependencies with: pip install ayn-ml[excel]"
                ) from last_err

        cols = required_columns(plan)
        if not cols:
            logger.warning(
                "ExcelSource: required_columns returned an empty list for plan %r — "
                "check that data_schema and metrics are configured correctly",
                plan.name,
            )
        nw_df = nw.from_native(native_df, eager_only=True)
        present = [c for c in cols if c in nw_df.columns]
        logger.debug(
            "ExcelSource: projecting %d/%d required columns from %s",
            len(present),
            len(cols),
            path,
        )
        return nw.to_native(nw_df.select(present))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_backend(self, backend: str, path: Path) -> Any:
        """Read *path* with the specified backend.

        ``sheet_name`` uses pandas conventions (0-based integer index).
        Integer values are translated to Polars' 1-based ``sheet_id``
        when the Polars backend is active.

        Args:
            backend: ``"polars"`` or ``"pandas"``.
            path: Resolved path to the Excel file.

        Returns:
            A native DataFrame (Polars or pandas).
        """
        if backend == "polars":
            try:
                import polars as pl
            except ImportError as exc:
                raise ImportError(
                    "Backend 'polars' requested but polars is not installed."
                ) from exc
            try:
                import fastexcel  # noqa: F401  # :UNSAFE: fastexcel optional; pip install ayn-ml[excel]
            except ImportError as exc:
                raise ImportError(
                    "fastexcel is required for the Polars Excel backend. "
                    "Install with: pip install ayn-ml[excel]"
                ) from exc
            # Polars uses sheet_name (str) or sheet_id (1-based int).
            if isinstance(self.sheet_name, str):
                return pl.read_excel(path, sheet_name=self.sheet_name, **self.read_kwargs)
            else:
                return pl.read_excel(path, sheet_id=self.sheet_name + 1, **self.read_kwargs)

        if backend == "pandas":
            try:
                import openpyxl  # noqa: F401  # :UNSAFE: openpyxl optional; pip install ayn-ml[excel]
            except ImportError as exc:
                raise ImportError(
                    "openpyxl is required for the pandas Excel backend. "
                    "Install with: pip install ayn-ml[excel]"
                ) from exc
            import pandas as pd
            return pd.read_excel(path, sheet_name=self.sheet_name, **self.read_kwargs)

        raise ValueError(
            f"Unsupported backend for ExcelSource: {backend!r}. "
            "Use 'polars' or 'pandas'."
        )
