"""Strategies by name."""

from __future__ import annotations

from typing import Any

from fincomb.exceptions import UnknownStrategyError
from fincomb.strategies.base import Strategy
from fincomb.strategies.buy_and_hold import BuyAndHold
from fincomb.strategies.fixed_weights import FixedWeights
from fincomb.strategies.mean_reversion import MeanReversion
from fincomb.strategies.momentum import Momentum
from fincomb.strategies.optimized import Optimized
from fincomb.strategies.params import json_schema

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (BuyAndHold, FixedWeights, Optimized, Momentum, MeanReversion)
}


def get_strategy(name: str) -> type[Strategy]:
    key = name.strip().lower().replace("-", "_")
    if key not in STRATEGIES:
        raise UnknownStrategyError(
            f"unknown strategy {name!r}; choose one of {', '.join(STRATEGIES)}"
        )
    return STRATEGIES[key]


def describe(cls: type[Strategy]) -> dict[str, Any]:
    return {
        "name": cls.name,
        "description": cls.description,
        "default_rebalance": cls.default_rebalance,
        "uses_prices": cls.uses_prices,
        "params": [
            {
                "name": s.name,
                "type": s.type,
                "default": s.default,
                "help": s.help,
                "minimum": s.minimum,
                "maximum": s.maximum,
                "choices": list(s.choices) if s.choices else None,
            }
            for s in cls.param_specs
        ],
        "params_schema": json_schema(cls.param_specs),
    }


def describe_all() -> list[dict[str, Any]]:
    return [describe(cls) for cls in STRATEGIES.values()]
