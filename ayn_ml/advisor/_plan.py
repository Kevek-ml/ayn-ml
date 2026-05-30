"""SuggestedPlan — output container for MetricAdvisor.suggest()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ayn_ml.core.spec import MonitoringPlan


@dataclass(frozen=True)
class SuggestedPlan:
    """Output of ``MetricAdvisor.suggest()``.

    Wraps a ready-to-use ``MonitoringPlan`` together with human-readable
    warnings that explain the advisor's decisions — which metrics were
    added, demoted, or excluded, and why.

    Args:
        plan: The generated ``MonitoringPlan``, ready to pass to ``Runner``.
        warnings: Ordered tuple of advisory messages.  Each message names the
            affected column or metric and gives the quantitative reason (e.g.
            ``"levene added for 'age': variance_ratio=1.83"``).

    Example::

        result = designer.suggest(df, reference=ref_df)
        plan   = result.plan
        for w in result.warnings:
            print(w)
    """

    plan: MonitoringPlan
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            Dict with ``"plan"`` (from ``MonitoringPlan.model_dump()``) and
            ``"warnings"`` (list of strings).
        """
        return {"plan": self.plan.model_dump(), "warnings": list(self.warnings)}
