"""Shared helpers for store implementations.

Private module — not part of the public API.  Provides the two
serialisation functions used by both InMemoryStore and SqliteStore to
produce consistent flat-dict rows from MonitoringReport objects.
"""

from __future__ import annotations

from typing import Any

from ayn_ml.core.result import ExecutionContext, MetricResult, MonitoringReport
from ayn_ml.core.spec import MonitoringPlan


def to_row(result: MetricResult, ctx: ExecutionContext) -> dict[str, Any]:
    """Flatten a MetricResult + ExecutionContext into a flat dict row.

    Args:
        result: A single metric result from a MonitoringReport.
        ctx: The ExecutionContext of the run that produced the result.

    Returns:
        Flat dict with keys: ``run_id``, ``model_id``, ``model_version``,
        ``metric_name``, ``feature_name``, ``value``, ``status``,
        ``effect_size``, ``effect_size_label``, ``period_start``,
        ``period_end``, ``metric_type``.
    """
    return {
        "run_id": ctx.run_id,
        "model_id": ctx.model_id,
        "model_version": ctx.model_version,
        "metric_name": result.spec.name,
        "feature_name": result.spec.feature_name,
        "value": result.value,
        "status": result.status,
        "effect_size": result.effect_size,
        "effect_size_label": result.effect_size_label,
        "period_start": ctx.period_start.isoformat() if ctx.period_start else None,
        "period_end": ctx.period_end.isoformat() if ctx.period_end else None,
        "metric_type": result.spec.metric_type.value if result.spec.metric_type else None,
    }


def profile_to_rows(report: MonitoringReport) -> list[dict[str, Any]]:
    """Flatten MonitoringReport.profile into metric_results-shaped dicts.

    Each (column, stat_name) pair in the profile becomes one row with
    ``metric_type = "profile"``.  Returns an empty list when
    ``report.profile`` is ``None``.

    Args:
        report: The MonitoringReport containing profile data.

    Returns:
        List of flat dicts, one per (column, stat_name) pair, using the
        same key set as ``to_row()`` with ``metric_type = "profile"`` and
        ``status``, ``effect_size``, ``effect_size_label`` set to ``None``.
    """
    if report.profile is None:
        return []
    ctx = report.context
    rows = []
    for col_name, stats in report.profile.items():
        for stat_name, stat_value in stats.items():
            rows.append(
                {
                    "run_id": ctx.run_id,
                    "model_id": ctx.model_id,
                    "model_version": ctx.model_version,
                    "metric_name": stat_name,
                    "feature_name": col_name,
                    "value": stat_value,
                    "status": None,
                    "effect_size": None,
                    "effect_size_label": None,
                    "period_start": ctx.period_start.isoformat() if ctx.period_start else None,
                    "period_end": ctx.period_end.isoformat() if ctx.period_end else None,
                    "metric_type": "profile",
                }
            )
    return rows


def extract_plan_meta(plan: MonitoringPlan, ctx: ExecutionContext) -> dict[str, Any]:
    """Extract a flat subset of MonitoringPlan + ExecutionContext fields.

    Prefixes plan-level fields with ``plan_`` and run-level fields with
    ``run_``.  Only a curated subset of plan fields is included — fields
    useful for grouping and filtering time-series queries.

    Args:
        plan: The MonitoringPlan that drove the run.
        ctx: The ExecutionContext of the run.

    Returns:
        Flat dict with keys: ``plan_name``, ``plan_window_type``,
        ``plan_window_n``, ``plan_sampling_type``, ``plan_sampling_frac``,
        ``run_n_current``, ``run_n_reference``.
    """
    window = plan.window
    sampling = plan.sampling
    return {
        "plan_name": plan.name,
        "plan_window_type": window.type if window else None,
        "plan_window_n": getattr(window, "n", None),
        "plan_sampling_type": sampling.type if sampling else None,
        "plan_sampling_frac": getattr(sampling, "frac", None),
        "run_n_current": ctx.n_current,
        "run_n_reference": ctx.n_reference,
    }


def report_to_rows(
    report: MonitoringReport,
    get_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Convert a MonitoringReport into a list of flat metric rows.

    Includes both metric result rows (from ``report.results``) and profile
    rows (from ``report.profile``, if present).

    Args:
        report: The MonitoringReport to flatten.
        get_metadata: When ``True``, enrich each row with plan and run
            metadata via ``extract_plan_meta()``.

    Returns:
        One dict per MetricResult in ``report.results``, followed by one
        dict per (column, stat) pair in ``report.profile`` (if any).
    """
    ctx = report.context
    meta = extract_plan_meta(report.plan, ctx) if get_metadata else {}
    rows = []
    for result in report.results:
        row = to_row(result, ctx)
        if get_metadata:
            row.update(meta)
        rows.append(row)
    for profile_row in profile_to_rows(report):
        if get_metadata:
            profile_row = {**profile_row, **meta}
        rows.append(profile_row)
    return rows
