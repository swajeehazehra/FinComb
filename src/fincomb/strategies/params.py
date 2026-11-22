"""Strategy parameters: one declaration drives parsing, validation and the JSON Schema.

The quant core does not depend on Pydantic, so each strategy declares its parameters
as `ParamSpec`s. `parse_params` fills defaults, converts strings (from the CLI) and
JSON numbers (from the API) to the declared type, and rejects unknown names, so a
typo never silently falls back to a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

ParamType = Literal["int", "float", "str", "weights"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: ParamType
    default: Any
    help: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


def _to_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError(f"{name} must be a whole number, got {value!r}")


def _to_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        out = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return out


def parse_weights(name: str, value: Any) -> dict[str, float] | None:
    """{symbol: weight} from a dict or from text like "SPY=0.6,TLT=0.4"."""
    if value is None:
        return None
    if isinstance(value, str):
        pairs: dict[str, Any] = {}
        for part in value.split(","):
            if not part.strip():
                continue
            symbol, sep, number = part.partition("=")
            if not sep:
                raise ValueError(f"{name}: write weights as SYMBOL=WEIGHT, e.g. SPY=0.6")
            key = symbol.strip().upper()
            if key in pairs:
                raise ValueError(f"{name}: {key} is listed twice")
            pairs[key] = number
        value = pairs
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must map symbols to weights")
    out: dict[str, float] = {}
    for symbol, w in value.items():
        key = str(symbol).strip().upper()
        if key in out:
            raise ValueError(f"{name}: {key} is listed twice")
        out[key] = _to_float(f"{name}[{key}]", w)
    return out


def _convert(spec: ParamSpec, value: Any) -> Any:
    if spec.type == "weights":
        return parse_weights(spec.name, value)
    if value is None:
        raise ValueError(f"{spec.name} must not be empty")
    if spec.type == "int":
        out: Any = _to_int(spec.name, value)
    elif spec.type == "float":
        out = _to_float(spec.name, value)
    else:
        out = str(value).strip().lower().replace("-", "_")
        if spec.choices and out not in spec.choices:
            raise ValueError(f"{spec.name} must be one of {', '.join(spec.choices)}")
        return out
    if spec.minimum is not None and out < spec.minimum:
        raise ValueError(f"{spec.name} must be at least {spec.minimum:g}, got {out:g}")
    if spec.maximum is not None and out > spec.maximum:
        raise ValueError(f"{spec.name} must be at most {spec.maximum:g}, got {out:g}")
    return out


def parse_params(specs: tuple[ParamSpec, ...], given: dict[str, Any] | None) -> dict[str, Any]:
    """Every declared parameter, from `given` or its default; unknown names raise."""
    given = dict(given or {})
    known = {s.name for s in specs}
    unknown = sorted(set(given) - known)
    if unknown:
        allowed = ", ".join(sorted(known)) or "none"
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)} (allowed: {allowed})")
    out = {}
    for spec in specs:
        value = given.get(spec.name, spec.default)
        out[spec.name] = None if value is None and spec.type == "weights" else _convert(spec, value)
    return out


def json_schema(specs: tuple[ParamSpec, ...]) -> dict[str, Any]:
    """JSON Schema of a strategy's `params` object (shown by GET /strategies)."""
    props: dict[str, Any] = {}
    for s in specs:
        if s.type == "weights":
            prop: dict[str, Any] = {
                "type": ["object", "null"],
                "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
            }
        else:
            prop = {"type": {"int": "integer", "float": "number", "str": "string"}[s.type]}
            if s.minimum is not None:
                prop["minimum"] = s.minimum
            if s.maximum is not None:
                prop["maximum"] = s.maximum
            if s.choices:
                prop["enum"] = list(s.choices)
        prop["default"] = s.default
        prop["description"] = s.help
        props[s.name] = prop
    return {"type": "object", "properties": props, "additionalProperties": False}
