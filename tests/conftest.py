from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ayn_ml.core.result import ExecutionContext
from ayn_ml.core.schema import AgentSchema, TabularSchema, TextSchema
from ayn_ml.core.spec import MetricSpec, MonitoringPlan

_RNG = np.random.default_rng(42)
_N = 300


# ── Schemas ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tabular_schema() -> TabularSchema:
    return TabularSchema()


@pytest.fixture(scope="session")
def text_schema() -> TextSchema:
    return TextSchema()


@pytest.fixture(scope="session")
def agent_schema() -> AgentSchema:
    return AgentSchema()


# ── DataFrames ─────────────────────────────────────────────────────────────────


def _make_binary_df(rng: np.random.Generator, age_mean: float) -> pd.DataFrame:
    n = _N
    return pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "y_pred": rng.integers(0, 2, n),
            "y_pred_proba": rng.uniform(0, 1, n),
            "age": rng.normal(age_mean, 10, n),
            "income": rng.normal(60_000, 15_000, n),
            "category": rng.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture(scope="session")
def df_reference() -> pd.DataFrame:
    return _make_binary_df(_RNG, age_mean=40.0)


@pytest.fixture(scope="session")
def df_current() -> pd.DataFrame:
    return _make_binary_df(_RNG, age_mean=50.0)


# ── Core types ─────────────────────────────────────────────────────────────────


@pytest.fixture
def execution_context() -> ExecutionContext:
    return ExecutionContext(
        model_id="model_a",
        model_version="1.0",
        eval_timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def accuracy_spec() -> MetricSpec:
    return MetricSpec(name="accuracy")


@pytest.fixture
def accuracy_spec_with_threshold() -> MetricSpec:
    return MetricSpec(name="accuracy", threshold=0.8, upper_bound=False)


@pytest.fixture
def basic_plan(tabular_schema, accuracy_spec) -> MonitoringPlan:
    return MonitoringPlan(
        name="test_plan",
        model_id="model_a",
        model_version="1.0",
        data_schema=tabular_schema,
        metrics=[accuracy_spec],
    )
