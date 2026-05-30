"""Utility functions for preparing recommender-system evaluation data.

These helpers are not metrics — they transform raw data into the interactions
table format expected by :mod:`ayn_ml.metrics.recsys`.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from ayn_ml.exceptions import SchemaError


def interactions_from_matrix(
    truth_matrix: Any,
    pred_matrix: Any | None = None,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    relevance_col: str = "relevance",
    score_col: str = "score",
) -> Any:
    """Convert one or two user×item matrices into a long interactions table.

    Recommender-system evaluation data often comes as a wide matrix where
    rows are users and columns are items.  This function melts such matrices
    into the long ``(user_id, item_id, relevance[, score])`` format expected
    by all metrics in :mod:`ayn_ml.metrics.recsys`.

    Args:
        truth_matrix: Ground-truth user×item DataFrame.  Rows are users,
            columns are items, cell values are relevance signals (binary
            ``0``/``1`` or graded).  Must be an eager pandas or Polars
            DataFrame — narwhals ``LazyFrame`` is not supported.  The row
            index (pandas) or first column (Polars) is used as the user
            identifier; column names are used as item identifiers.
        pred_matrix: Optional predicted scores user×item DataFrame.  Must
            have the same shape, row labels, and column names as
            ``truth_matrix``.  Cell values are predicted scores.  When
            ``None``, no ``score_col`` column is added to the output and
            items are taken in their DataFrame column order when ranking.
        user_col: Name of the user column in the output table.
        item_col: Name of the item column in the output table.
        relevance_col: Name of the ground-truth relevance column in the
            output table.
        score_col: Name of the predicted score column in the output table.
            Ignored when ``pred_matrix`` is ``None``.

    Returns:
        Long-format DataFrame in the same library as ``truth_matrix``
        (pandas or Polars) with columns ``[user_col, item_col,
        relevance_col]`` plus ``score_col`` when ``pred_matrix`` is
        provided.

    Raises:
        SchemaError: If ``pred_matrix`` is provided but its shape or
            labels do not match ``truth_matrix``.
        TypeError: If ``truth_matrix`` is not a supported eager DataFrame.

    Examples:
        >>> import pandas as pd
        >>> truth = pd.DataFrame(
        ...     {"item_a": [1, 0], "item_b": [0, 1]},
        ...     index=["user_1", "user_2"],
        ... )
        >>> interactions_from_matrix(truth)
           user_id  item_id  relevance
        0  user_1   item_a          1
        1  user_1   item_b          0
        2  user_2   item_a          0
        3  user_2   item_b          1
    """
    truth_native = nw.from_native(truth_matrix, eager_only=True)

    # Detect library and extract user labels + item names
    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(truth_matrix, pd.DataFrame):
            return _from_matrix_pandas(
                truth_matrix,
                pred_matrix,
                user_col=user_col,
                item_col=item_col,
                relevance_col=relevance_col,
                score_col=score_col,
            )
    except ImportError:
        pass

    try:
        import polars as pl  # noqa: PLC0415

        if isinstance(truth_matrix, pl.DataFrame):
            return _from_matrix_polars(
                truth_matrix,
                pred_matrix,
                user_col=user_col,
                item_col=item_col,
                relevance_col=relevance_col,
                score_col=score_col,
            )
    except ImportError:
        pass

    # nw.from_native above already rejected unsupported types (LazyFrame, list, etc.)
    _ = truth_native  # suppress lint; the assignment serves as the unsupported-type guard
    raise TypeError(
        f"truth_matrix must be a pandas or Polars eager DataFrame, got {type(truth_matrix).__name__}."
    )


def _from_matrix_pandas(
    truth: Any,
    pred: Any | None,
    *,
    user_col: str,
    item_col: str,
    relevance_col: str,
    score_col: str,
) -> Any:
    """Pandas implementation of interactions_from_matrix.

    .. note::
        # :UNSAFE: uses native pandas melt/merge; narwhals unpivot cannot represent
        # the pandas row-index-as-user-id pattern required by this utility
    """
    import pandas as pd  # noqa: PLC0415

    if pred is not None:
        if truth.shape != pred.shape:
            raise SchemaError(
                f"truth_matrix shape {truth.shape} does not match "
                f"pred_matrix shape {pred.shape}."
            )
        if list(truth.columns) != list(pred.columns):
            raise SchemaError("truth_matrix and pred_matrix must have identical column names.")
        if list(truth.index) != list(pred.index):
            raise SchemaError("truth_matrix and pred_matrix must have identical row indices.")

    def _melt(df: Any, value_name: str) -> Any:
        # rename_axis ensures reset_index always produces a column named "__user__",
        # regardless of whether the index has a name or not.
        return (
            df.rename_axis("__user__")
            .reset_index()
            .rename(columns={"__user__": user_col})
            .melt(id_vars=user_col, var_name=item_col, value_name=value_name)
        )

    truth_long = _melt(truth, relevance_col)

    if pred is None:
        return truth_long.reset_index(drop=True)

    pred_long = _melt(pred, score_col)
    result = pd.merge(truth_long, pred_long[[user_col, item_col, score_col]], on=[user_col, item_col])
    return result.reset_index(drop=True)


def _from_matrix_polars(
    truth: Any,
    pred: Any | None,
    *,
    user_col: str,
    item_col: str,
    relevance_col: str,
    score_col: str,
) -> Any:
    """Polars implementation of interactions_from_matrix.

    .. note::
        # :UNSAFE: uses native polars unpivot/join; narwhals cannot represent
        # the positional-user-index path required when no user column is present
    """
    import polars as pl  # noqa: PLC0415

    # In Polars, the user IDs must be in a dedicated column.
    # Convention: if the first column is named "user_id" (or user_col), use it;
    # otherwise assume rows are indexed by position and add a user index.
    cols = truth.columns
    if cols[0] == user_col:
        user_series = truth[user_col]
        item_cols = cols[1:]
        truth_items = truth.select(item_cols)
        pred_items = pred.select(pred.columns[1:]) if pred is not None else None
    else:
        # No user column — use row index as user label
        user_series = pl.Series(user_col, list(range(truth.height)))
        item_cols = cols
        truth_items = truth
        pred_items = pred

    if pred is not None:
        if truth_items.shape != pred_items.shape:
            raise SchemaError(
                f"truth_matrix shape {truth_items.shape} does not match "
                f"pred_matrix shape {pred_items.shape}."
            )
        if list(truth_items.columns) != list(pred_items.columns):
            raise SchemaError("truth_matrix and pred_matrix must have identical column names.")

    # Build long format by unpivoting
    truth_with_user = truth_items.with_columns(user_series)
    truth_long = truth_with_user.unpivot(
        index=user_col, variable_name=item_col, value_name=relevance_col
    )

    if pred_items is None:
        return truth_long

    pred_with_user = pred_items.with_columns(user_series)
    pred_long = pred_with_user.unpivot(
        index=user_col, variable_name=item_col, value_name=score_col
    )
    return truth_long.join(pred_long.select([user_col, item_col, score_col]), on=[user_col, item_col])
