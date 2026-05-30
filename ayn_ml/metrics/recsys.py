"""Recommender-system metrics.

All recsys metrics operate on an interactions DataFrame — one row per
(user, item) pair — described by a :class:`~ayn_ml.core.schema.RecSysSchema`.

Metric params are passed via ``MetricSpec.params``:

- ``k`` (int, default 10): ranking cutoff.
- ``relevance_threshold`` (float, default 0): items with
  ``relevance > threshold`` are considered relevant.
- ``beta`` (float, default 1.0): F-beta weight (FBetaAtK only).
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from ayn_ml.core.result import MetricResult
from ayn_ml.core.schema import DataSchema, RecSysSchema
from ayn_ml.core.spec import MetricSpec, MetricType
from ayn_ml.exceptions import InsufficientDataError, SchemaError
from ayn_ml.metrics.base import compute_status
from ayn_ml.metrics.registry import register_metric

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_MIN_USERS: int = 1


def _result_recsys(value: float, spec: MetricSpec) -> MetricResult:
    """Build a MetricResult rounded to 6 decimal places with pass/fail status."""
    rounded = round(value, 6)
    return MetricResult(spec=spec, value=rounded, status=compute_status(rounded, spec))


def _check_recsys(schema: DataSchema) -> RecSysSchema:
    """Assert ``schema`` is a RecSysSchema and return it; raise SchemaError otherwise."""
    if not isinstance(schema, RecSysSchema):
        raise SchemaError(f"Expected RecSysSchema, got {type(schema).__name__}.")
    return schema


def _get_k(spec: MetricSpec) -> int:
    """Return the ranking cutoff ``k`` from ``spec.params`` (default 10); reject non-positive or bool values."""
    k = spec.params.get("k", 10)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"params['k'] must be a positive integer, got {k!r}.")
    return k


def _get_relevance_threshold(spec: MetricSpec) -> float:
    """Return the relevance threshold from ``spec.params`` (default 0)."""
    return float(spec.params.get("relevance_threshold", 0))


def _build_user_rankings(
    df: Any,
    schema: RecSysSchema,
    k: int,
    relevance_threshold: float,
) -> list[tuple[set, list]]:
    """Return per-user ``(relevant_items, top_k_predicted)`` pairs.

    Args:
        df: Interactions DataFrame — one row per (user, item) interaction.
        schema: RecSysSchema describing column roles.
        k: Number of items to include in the predicted ranking.
        relevance_threshold: Items with ``relevance > threshold`` are relevant.

    Returns:
        List of ``(relevant_set, top_k_list)`` tuples, one per unique user.

    Raises:
        SchemaError: If a required column is absent from ``df``.
        InsufficientDataError: If ``df`` contains no users.
    """
    native = nw.from_native(df, eager_only=True)

    for col in (schema.user_id_col, schema.item_id_col, schema.relevance_col):
        if col not in native.columns:
            raise SchemaError(f"Column '{col}' not found in the evaluation data.")

    user_ids = native[schema.user_id_col].to_numpy()
    item_ids = native[schema.item_id_col].to_numpy()
    relevance = native[schema.relevance_col].to_numpy().astype(float)

    has_scores = schema.score_col is not None and schema.score_col in native.columns
    scores: np.ndarray | None = None
    rank_mode = False
    if has_scores:
        scores = native[schema.score_col].to_numpy().astype(float)
        rank_mode = schema.recommendations_type == "rank"

    unique_users = np.unique(user_ids)
    if len(unique_users) < _MIN_USERS:
        raise InsufficientDataError("At least 1 user is required in the evaluation data.")

    # :PERF: numpy groupby loop; narwhals group_by cannot collect items into lists without pyarrow
    rankings: list[tuple[set, list]] = []
    for user in unique_users:
        mask = user_ids == user
        u_items = item_ids[mask]
        u_rel = relevance[mask]
        relevant: set = set(u_items[u_rel > relevance_threshold].tolist())

        if scores is not None:
            u_scores = scores[mask]
            order = np.argsort(u_scores) if rank_mode else np.argsort(-u_scores)
            top_k = u_items[order[:k]].tolist()
        else:
            top_k = u_items[:k].tolist()

        rankings.append((relevant, top_k))

    return rankings


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@register_metric("precision_at_k")
class PrecisionAtKMetric:
    """Mean Precision@K across all users.

    For each user: ``|relevant ∩ top_k| / K``.  Averaged across users.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.
    """

    name = "precision_at_k"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean Precision@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)
        per_user = [len(relevant & set(top_k)) / k for relevant, top_k in rankings]
        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


@register_metric("recall_at_k")
class RecallAtKMetric:
    """Mean Recall@K across all users.

    For each user: ``|relevant ∩ top_k| / |relevant|``.  Users with no
    relevant items contribute 0 to the average.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.
    """

    name = "recall_at_k"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean Recall@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)
        per_user = [
            len(relevant & set(top_k)) / len(relevant) if relevant else 0.0
            for relevant, top_k in rankings
        ]
        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


@register_metric("fbeta_at_k")
class FBetaAtKMetric:
    """Mean F-beta@K across all users.

    Harmonic mean of Precision@K and Recall@K weighted by ``beta``.
    ``beta=1`` gives the standard F1 score.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        beta (float): Weight parameter; default 1.0.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.
    """

    name = "fbeta_at_k"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean F-beta@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]``, ``params["beta"]``,
                and ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        beta = float(spec.params.get("beta", 1.0))
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)

        per_user = []
        for relevant, top_k in rankings:
            if not top_k:
                per_user.append(0.0)
                continue
            hits = len(relevant & set(top_k))
            p = hits / k
            r = hits / len(relevant) if relevant else 0.0
            denom = beta**2 * p + r
            per_user.append((1 + beta**2) * p * r / denom if denom > 0 else 0.0)

        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


@register_metric("hit_rate")
class HitRateMetric:
    """Mean Hit Rate@K across all users.

    Fraction of users for whom at least one relevant item appears in the
    top-K recommendations.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.
    """

    name = "hit_rate"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean Hit Rate@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)
        per_user = [1.0 if (relevant & set(top_k)) else 0.0 for relevant, top_k in rankings]
        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


@register_metric("map_at_k")
class MAPAtKMetric:
    """Mean Average Precision@K across all users (MAP@K).

    For each user, the Average Precision@K is the mean of Precision@i for
    every rank position ``i`` (1-based, up to ``k``) at which a relevant item
    appears.  Users with no relevant items contribute 0.

    ``AP@K = (1 / |relevant|) * Σ P@i * rel(i)``

    where ``rel(i)`` is 1 if the item at rank ``i`` is relevant, else 0.

    Note: normalization uses the full relevant-set size ``|relevant|``, not
    ``min(|relevant|, k)``.  When a user has more relevant items than ``k``,
    a perfect top-K list still yields ``AP@K < 1.0``.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.
    """

    name = "map_at_k"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute MAP@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)

        per_user = []
        for relevant, top_k in rankings:
            if not relevant:
                per_user.append(0.0)
                continue
            hits = 0
            precision_sum = 0.0
            for i, item in enumerate(top_k, start=1):
                if item in relevant:
                    hits += 1
                    precision_sum += hits / i
            per_user.append(precision_sum / len(relevant))

        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


@register_metric("ndcg_at_k")
class NDCGAtKMetric:
    """Normalized Discounted Cumulative Gain@K (NDCG@K).

    Measures ranking quality by discounting the gain of each relevant item
    by its position.  Normalized against the ideal DCG (items sorted by
    relevance descending).  Supports both binary and graded relevance.

    ``DCG@K = Σ rel(i) / log₂(i + 1)``   for i = 1 … k

    ``NDCG@K = DCG@K / IDCG@K``

    For binary relevance ``rel(i) ∈ {0, 1}``.  For graded relevance the
    raw ``relevance`` column values are used directly as gains.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.  Used only to build the
            relevant-item set for IDCG normalization; the raw relevance
            values are always used as gains.
    """

    name = "ndcg_at_k"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute NDCG@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)

        # NOTE: cannot reuse _build_user_rankings here — NDCG requires per-item gain
        # values that the helper discards.  Keep this extraction in sync with that helper
        # if schema handling or error messages change.
        native = nw.from_native(current, eager_only=True)
        for col in (rs.user_id_col, rs.item_id_col, rs.relevance_col):
            if col not in native.columns:
                raise SchemaError(f"Column '{col}' not found in the evaluation data.")

        user_ids = native[rs.user_id_col].to_numpy()
        item_ids = native[rs.item_id_col].to_numpy()
        relevance = native[rs.relevance_col].to_numpy().astype(float)

        has_scores = rs.score_col is not None and rs.score_col in native.columns
        scores: np.ndarray | None = None
        rank_mode = False
        if has_scores:
            scores = native[rs.score_col].to_numpy().astype(float)
            rank_mode = rs.recommendations_type == "rank"

        unique_users = np.unique(user_ids)
        if len(unique_users) < _MIN_USERS:
            raise InsufficientDataError("At least 1 user is required in the evaluation data.")

        # :PERF: numpy groupby loop; narwhals group_by cannot collect items into lists without pyarrow
        per_user = []
        for user in unique_users:
            mask = user_ids == user
            u_items = item_ids[mask]
            u_rel = relevance[mask]

            # Build item → relevance mapping for gain lookup
            item_gain: dict = dict(zip(u_items.tolist(), u_rel.tolist()))

            # Predicted top-k
            if scores is not None:
                u_scores = scores[mask]
                order = np.argsort(u_scores) if rank_mode else np.argsort(-u_scores)
                top_k = u_items[order[:k]].tolist()
            else:
                top_k = u_items[:k].tolist()

            # DCG
            dcg = sum(
                item_gain.get(item, 0.0) / np.log2(i + 2)
                for i, item in enumerate(top_k)
            )

            # IDCG — ideal ranking: sort all items by relevance descending, take top-k
            ideal_gains = sorted(u_rel.tolist(), reverse=True)[:k]
            idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))

            per_user.append(dcg / idcg if idcg > 0 else 0.0)

        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


@register_metric("mrr_at_k")
class MRRAtKMetric:
    """Mean Reciprocal Rank@K (MRR@K).

    For each user, the reciprocal rank is ``1 / rank`` where ``rank`` is the
    1-based position of the **first** relevant item in the top-K list.  If no
    relevant item appears in the top-K, the contribution is 0.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.
    """

    name = "mrr_at_k"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute MRR@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)

        per_user = []
        for relevant, top_k in rankings:
            rr = 0.0
            for i, item in enumerate(top_k, start=1):
                if item in relevant:
                    rr = 1.0 / i
                    break
            per_user.append(rr)

        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


# ---------------------------------------------------------------------------
# Beyond-accuracy helpers
# ---------------------------------------------------------------------------


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array.  Returns 0 for empty or all-zero input."""
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    index = np.arange(1, n + 1)
    return float((2.0 * np.dot(index, sorted_v)) / (n * sorted_v.sum()) - (n + 1) / n)


def _item_popularity_map(reference: Any, schema: RecSysSchema) -> tuple[dict, int]:
    """Build item → relative popularity from a training interactions DataFrame.

    Args:
        reference: Training interactions DataFrame (one row per interaction).
        schema: RecSysSchema; ``item_id_col`` identifies the item column.

    Returns:
        Tuple of (pop_map, n_ref) where ``pop_map`` maps item ID → fraction of
        total training interactions and ``n_ref`` is the total interaction count.

    Raises:
        SchemaError: If ``item_id_col`` is absent from ``reference``.
        InsufficientDataError: If ``reference`` is empty.
    """
    ref_native = nw.from_native(reference, eager_only=True)
    if schema.item_id_col not in ref_native.columns:
        raise SchemaError(f"Column '{schema.item_id_col}' not found in reference data.")
    ref_items = ref_native[schema.item_id_col].to_numpy()
    total = len(ref_items)
    if total == 0:
        raise InsufficientDataError("Reference data is empty.")
    unique, counts = np.unique(ref_items, return_counts=True)
    return dict(zip(unique.tolist(), (counts / total).tolist())), total


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------


@register_metric("diversity")
class DiversityMetric:
    """Mean intra-list diversity@K across all users.

    For each user, computes the mean pairwise cosine distance between the
    feature vectors of items in the top-K list, then averages across users.
    A value of 0 means all recommended items are identical; 1 means all pairs
    are orthogonal.

    Item feature vectors are read from columns in ``current`` listed in
    ``spec.params["item_features"]``.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        item_features (list[str]): Column names of numeric item features
            present in ``current``.  At least one column is required.
        relevance_threshold (float): Default 0.
    """

    name = "diversity"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean intra-list diversity@K.

        Args:
            current: Interactions DataFrame; must include the columns listed
                in ``params["item_features"]``.
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; requires ``params["item_features"]`` (list of
                column names); supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema, a required
                column is absent, or ``item_features`` is empty.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        feature_cols: list[str] = list(spec.params.get("item_features", []))
        if not feature_cols:
            raise SchemaError("params['item_features'] must list at least one feature column.")

        native = nw.from_native(current, eager_only=True)
        for col in feature_cols:
            if col not in native.columns:
                raise SchemaError(f"Item feature column '{col}' not found in current data.")

        # Build item → feature vector (first occurrence per item; features are item properties)
        # :PERF: numpy dedup; narwhals group_by+agg cannot produce row-level arrays without pyarrow
        item_ids_arr = native[rs.item_id_col].to_numpy()
        feat_arr = np.stack([native[c].to_numpy().astype(float) for c in feature_cols], axis=1)
        _, first_idx = np.unique(item_ids_arr, return_index=True)
        item_feature_map: dict = {item_ids_arr[i]: feat_arr[i] for i in first_idx}

        rankings = _build_user_rankings(current, rs, k, threshold)

        per_user = []
        for _, top_k in rankings:
            vecs = np.array(
                [item_feature_map[item] for item in top_k if item in item_feature_map],
                dtype=float,
            )
            if len(vecs) < 2:
                per_user.append(0.0)
                continue
            # Filter zero-norm vectors — they have no direction and would
            # corrupt the cosine similarity computation.
            norms = np.linalg.norm(vecs, axis=1)
            vecs = vecs[norms > 0]
            if len(vecs) < 2:
                per_user.append(0.0)
                continue
            norms_valid = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs_norm = vecs / norms_valid
            cos_sim = vecs_norm @ vecs_norm.T
            n = len(vecs)
            mean_sim = (cos_sim.sum() - n) / (n * (n - 1))
            per_user.append(1.0 - float(mean_sim))

        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)


@register_metric("novelty")
class NoveltyMetric:
    """Mean novelty of top-K recommendations across all users.

    For each recommended item: ``novelty(i) = -log₂(pop(i))`` where
    ``pop(i)`` is the fraction of training interactions involving item ``i``.
    Higher values indicate less popular (more novel) recommendations.

    Items absent from the training data receive the minimum possible
    popularity ``1 / N_ref`` (treated as a single interaction).

    Requires ``reference`` = training interactions DataFrame (same schema
    as ``current``; only ``item_id_col`` is used).

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Default 0.
    """

    name = "novelty"
    metric_type = MetricType.recsys
    requires_reference = True
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean novelty@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Training interactions DataFrame used to estimate item
                popularity.  Must not be ``None``.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a non-negative float value (unbounded above).

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema, ``reference``
                is ``None``, or a required column is absent.
            InsufficientDataError: If ``current`` or ``reference`` is empty.
        """
        rs = _check_recsys(schema)
        if reference is None:
            raise SchemaError("novelty requires reference data (training interactions).")
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)

        pop_map, n_ref = _item_popularity_map(reference, rs)
        min_pop = 1.0 / n_ref

        rankings = _build_user_rankings(current, rs, k, threshold)
        scores = [
            -np.log2(pop_map.get(item, min_pop))
            for _, top_k in rankings
            for item in top_k
        ]
        return _result_recsys(sum(scores) / len(scores) if scores else 0.0, spec)


@register_metric("popularity_bias")
class PopularityBiasMetric:
    """Mean popularity of top-K recommended items across all users.

    For each recommended item, its popularity is the fraction of training
    interactions involving that item.  Higher values indicate a bias toward
    recommending widely-seen items.

    Requires ``reference`` = training interactions DataFrame (same schema
    as ``current``; only ``item_id_col`` is used).

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Default 0.
    """

    name = "popularity_bias"
    metric_type = MetricType.recsys
    requires_reference = True
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean popularity bias@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Training interactions DataFrame used to estimate item
                popularity.  Must not be ``None``.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema, ``reference``
                is ``None``, or a required column is absent.
            InsufficientDataError: If ``current`` or ``reference`` is empty.
        """
        rs = _check_recsys(schema)
        if reference is None:
            raise SchemaError("popularity_bias requires reference data (training interactions).")
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)

        pop_map, _ = _item_popularity_map(reference, rs)
        rankings = _build_user_rankings(current, rs, k, threshold)
        scores = [
            pop_map.get(item, 0.0)
            for _, top_k in rankings
            for item in top_k
        ]
        return _result_recsys(sum(scores) / len(scores) if scores else 0.0, spec)


@register_metric("personalization")
class PersonalizationMetric:
    """Mean inter-user dissimilarity of top-K recommendation lists.

    Measures how different users' recommendation lists are from each other.
    Each user's top-K list is represented as a binary indicator vector over
    all recommended items; the metric is 1 minus the mean pairwise cosine
    similarity across all user pairs.

    Range ``[0, 1]``: 0 = every user gets an identical list; 1 = all lists
    are completely disjoint.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Default 0.
    """

    name = "personalization"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute personalization@K.

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.  Returns 0 when
            fewer than 2 users are present.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)

        if len(rankings) < 2:
            return _result_recsys(0.0, spec)

        all_items = sorted({item for _, top_k in rankings for item in top_k})
        item_idx = {item: i for i, item in enumerate(all_items)}
        n_items = len(item_idx)
        n_users = len(rankings)

        # :PERF: numpy dense matrix; acceptable for monitoring batch sizes
        mat = np.zeros((n_users, n_items), dtype=float)
        for u, (_, top_k) in enumerate(rankings):
            for item in top_k:
                mat[u, item_idx[item]] = 1.0

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat_norm = mat / norms
        cos_sim = mat_norm @ mat_norm.T
        mean_sim = (cos_sim.sum() - n_users) / (n_users * (n_users - 1))
        return _result_recsys(1.0 - float(mean_sim), spec)


@register_metric("item_bias")
class ItemBiasMetric:
    """Gini coefficient of item recommendation frequency.

    Measures how concentrated recommendations are on a small set of items.
    Counts how many times each unique item appears across all users' top-K
    lists and computes the Gini coefficient of those counts.

    Range ``[0, 1)``: 0 = all items recommended equally often; values
    approaching 1 indicate a few items dominate all recommendation slots.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Default 0.
    """

    name = "item_bias"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute item bias (Gini of item recommendation frequency).

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1)``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)

        item_counts: dict = {}
        for _, top_k in rankings:
            for item in top_k:
                item_counts[item] = item_counts.get(item, 0) + 1

        counts = np.array(list(item_counts.values()), dtype=float)
        return _result_recsys(_gini(counts), spec)


@register_metric("user_bias")
class UserBiasMetric:
    """Gini coefficient of per-user recommendation list length.

    Measures whether some users systematically receive shorter recommendation
    lists than others.  This occurs when a user's item catalogue is smaller
    than ``k`` (cold-start, sparse data).

    Range ``[0, 1)``: 0 = all users receive lists of equal length; higher
    values indicate imbalance in recommendation coverage across users.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        relevance_threshold (float): Default 0.
    """

    name = "user_bias"
    metric_type = MetricType.recsys
    requires_reference = False
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute user bias (Gini of per-user recommendation list length).

        Args:
            current: Interactions DataFrame (one row per user-item pair).
            reference: Ignored.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; supports ``params["k"]`` and
                ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1)``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema or a required
                column is absent.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        rankings = _build_user_rankings(current, rs, k, threshold)

        lengths = np.array([len(top_k) for _, top_k in rankings], dtype=float)
        return _result_recsys(_gini(lengths), spec)


@register_metric("serendipity")
class SerendipityMetric:
    """Mean serendipity@K across all users (deterministic variant).

    Serendipity measures how surprising *and* relevant the recommendations
    are.  For each item in the top-K list:

    ``serendipity(i, u) = relevance(i, u) × unexpectedness(i, u)``

    where ``unexpectedness(i, u)`` is the cosine distance between item
    ``i``'s feature vector and the centroid of item features the user
    interacted with in the training data (``reference``).  Items already
    familiar to the user score 0 unexpectedness; items with orthogonal
    features score 1.

    The metric is averaged over the top-K items for each user, then across
    users.  Range ``[0, 1]``.

    Requires:
        - ``reference``: training interactions DataFrame (same schema as
          ``current``; ``item_id_col`` is used to build per-user profiles).
        - ``params["item_features"]``: list of numeric feature column names
          present in ``current`` used to compute item vectors.

    Params (via ``spec.params``):
        k (int): Ranking cutoff; default 10.
        item_features (list[str]): Column names of numeric item features
            in ``current``.  At least one column is required.
        relevance_threshold (float): Items with ``relevance > threshold``
            are considered relevant; default 0.
    """

    name = "serendipity"
    metric_type = MetricType.recsys
    requires_reference = True
    accepted_column_types: frozenset = frozenset()

    def compute(
        self, current: Any, reference: Any | None, schema: DataSchema, spec: MetricSpec
    ) -> MetricResult:
        """Compute mean serendipity@K.

        Args:
            current: Interactions DataFrame; must include item feature
                columns listed in ``params["item_features"]``.
            reference: Training interactions DataFrame used to build
                per-user familiarity profiles.  Must not be ``None``.
            schema: RecSysSchema describing column roles.
            spec: MetricSpec; requires ``params["item_features"]``; supports
                ``params["k"]`` and ``params["relevance_threshold"]``.

        Returns:
            MetricResult with a float value in ``[0, 1]``.

        Raises:
            SchemaError: If ``schema`` is not a RecSysSchema, ``reference``
                is ``None``, a required column is absent, or
                ``item_features`` is empty.
            InsufficientDataError: If ``current`` contains no users.
        """
        rs = _check_recsys(schema)
        if reference is None:
            raise SchemaError("serendipity requires reference data (training interactions).")
        k = _get_k(spec)
        threshold = _get_relevance_threshold(spec)
        feature_cols: list[str] = list(spec.params.get("item_features", []))
        if not feature_cols:
            raise SchemaError("params['item_features'] must list at least one feature column.")

        native = nw.from_native(current, eager_only=True)
        for col in feature_cols:
            if col not in native.columns:
                raise SchemaError(f"Item feature column '{col}' not found in current data.")

        # Build item → feature vector map from current (first occurrence per item)
        # :PERF: numpy dedup; narwhals group_by+agg cannot produce row-level arrays without pyarrow
        item_ids_arr = native[rs.item_id_col].to_numpy()
        feat_arr = np.stack([native[col].to_numpy().astype(float) for col in feature_cols], axis=1)
        _, first_idx = np.unique(item_ids_arr, return_index=True)
        item_feature_map: dict = {item_ids_arr[i]: feat_arr[i] for i in first_idx}

        # Build per-user training profile: centroid of feature vectors of training items
        # :PERF: numpy groupby loop; narwhals group_by cannot collect items into lists without pyarrow
        ref_native = nw.from_native(reference, eager_only=True)
        if rs.user_id_col not in ref_native.columns or rs.item_id_col not in ref_native.columns:
            raise SchemaError("reference must contain user_id_col and item_id_col columns.")
        ref_users = ref_native[rs.user_id_col].to_numpy()
        ref_items = ref_native[rs.item_id_col].to_numpy()

        user_profile: dict = {}
        for user in np.unique(ref_users):
            mask = ref_users == user
            vecs = np.array(
                [item_feature_map[item] for item in ref_items[mask].tolist() if item in item_feature_map],
                dtype=float,
            )
            if len(vecs) == 0:
                continue
            # Filter zero-norm vectors before computing centroid
            norms = np.linalg.norm(vecs, axis=1)
            vecs = vecs[norms > 0]
            if len(vecs) == 0:
                continue
            centroid = vecs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            user_profile[user] = centroid / c_norm if c_norm > 0 else None

        rankings = _build_user_rankings(current, rs, k, threshold)
        # Rebuild user list in the same order as rankings (np.unique → sorted)
        cur_users = np.unique(native[rs.user_id_col].to_numpy())

        per_user = []
        for user, (relevant, top_k) in zip(cur_users, rankings):
            profile = user_profile.get(user)
            item_scores = []
            for item in top_k:
                rel = 1.0 if item in relevant else 0.0
                if rel == 0.0 or item not in item_feature_map:
                    item_scores.append(0.0)
                    continue
                vec = item_feature_map[item]
                v_norm = np.linalg.norm(vec)
                if v_norm == 0 or profile is None:
                    # Unknown profile → max unexpectedness (1.0) for relevant items
                    unexpectedness = 1.0
                else:
                    unexpectedness = float(1.0 - np.dot(vec / v_norm, profile))
                    unexpectedness = max(0.0, min(1.0, unexpectedness))
                item_scores.append(rel * unexpectedness)
            per_user.append(sum(item_scores) / len(item_scores) if item_scores else 0.0)

        return _result_recsys(sum(per_user) / len(per_user) if per_user else 0.0, spec)
