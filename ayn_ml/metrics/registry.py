"""Metric registry: registration decorator and lookup helpers.

The registry is a module-level dict mapping metric names to their
implementation classes.  Metric files register themselves at import time via
the ``@register_metric`` decorator; the tabular ``__init__.py`` triggers those
imports so the registry is fully populated before any lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ayn_ml.exceptions import UnknownMetricError

if TYPE_CHECKING:
    from ayn_ml.core.spec import MetricSpec, MetricType
    from ayn_ml.metrics.base import Metric

_REGISTRY: dict[str, type] = {}
"""Internal name → class mapping.  Populated by ``@register_metric``."""


def register_metric(name: str) -> Callable[[type], type]:
    """Class decorator that registers a metric implementation by name.

    Args:
        name: Registry key (e.g. ``"accuracy"``).  Must be unique across the
            entire library; re-registering the same name raises immediately.

    Returns:
        A decorator that registers the decorated class and returns it unchanged.

    Raises:
        ValueError: If ``name`` is already present in the registry.

    Example:
        @register_metric("my_metric")
        class MyMetric:
            ...
    """

    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_metric(name: str) -> Metric:
    """Look up and instantiate a metric by its registry name.

    A fresh instance is returned on every call so metrics remain stateless
    across concurrent runs.

    Args:
        name: Registry key of the desired metric.

    Returns:
        A new instance of the corresponding metric class.

    Raises:
        UnknownMetricError: If ``name`` is not found in the registry.  The
            error message includes the sorted list of available names.
    """
    if name not in _REGISTRY:
        raise UnknownMetricError(f"Metric '{name}' not found. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_metrics() -> list[str]:
    """Return a sorted list of all registered metric names.

    Returns:
        Alphabetically sorted list of registry keys.
    """
    return sorted(_REGISTRY.keys())


def resolve_metric_type(spec: MetricSpec) -> MetricType:
    """Return the effective MetricType for a spec.

    When ``spec.metric_type`` is not ``None``, it is returned as-is.
    Otherwise the type is read from the registered class's ``metric_type``
    class attribute, avoiding the need to specify it explicitly in
    ``MetricSpec``.

    Args:
        spec: MetricSpec whose type should be resolved.

    Returns:
        The resolved MetricType.

    Raises:
        UnknownMetricError: If ``spec.metric_type`` is ``None`` and
            ``spec.name`` is not present in the registry.
        AttributeError: If the registered class has no ``metric_type``
            attribute (should not happen for built-in metrics).
    """
    if spec.metric_type is not None:
        return spec.metric_type
    if spec.name not in _REGISTRY:
        raise UnknownMetricError(
            f"Cannot infer metric_type: '{spec.name}' not found in registry. "
            f"Pass metric_type explicitly or register the metric first."
        )
    cls = _REGISTRY[spec.name]
    try:
        return cls.metric_type
    except AttributeError:
        raise AttributeError(
            f"Metric '{spec.name}' is registered but missing a 'metric_type' class attribute — this is a library bug."
        ) from None
