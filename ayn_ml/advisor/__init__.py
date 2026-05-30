"""ayn_ml.advisor — automatic MonitoringPlan generation from data characteristics.

Public exports: ``MetricAdvisor``, ``SuggestedPlan``.

Example::

    from ayn_ml.advisor import MetricAdvisor, SuggestedPlan
    from ayn_ml.core.schema import TabularSchema

    schema = TabularSchema(label_col="y_true", prediction_col="y_pred")
    designer = MetricAdvisor(schema)
    result: SuggestedPlan = designer.suggest(df, reference=ref_df)
    plan = result.plan
"""

from ayn_ml.advisor._analysis import ColumnAnalysis
from ayn_ml.advisor._plan import SuggestedPlan
from ayn_ml.advisor.advisor import MetricAdvisor

__all__ = ["ColumnAnalysis", "MetricAdvisor", "SuggestedPlan"]
