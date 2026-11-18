"""A small wrapper around SciPy's SLSQP for long-only, fully invested portfolios.

`solve` minimises a smooth objective subject to sum(w) = 1 and per-asset bounds,
from one or more feasible starting points, and only returns weights that pass an
independent check (finite, sum to 1, inside the bounds). Everything is
deterministic: the same inputs always give the same weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

FTOL = 1e-12
MAX_ITER = 500
BOUND_SNAP = 1e-10  # weights this close to a bound are put exactly on it
CHECK_TOL = 1e-8  # tolerance for the sum and bound checks

Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SolveResult:
    weights: np.ndarray | None  # None: no start produced valid weights
    success: bool
    message: str
    iterations: int  # of the run that was kept (or the last run on failure)
    objective: float


def clean_weights(w: np.ndarray, lower: float, upper: float) -> np.ndarray | None:
    """Snap rounding dust to the bounds, renormalise, and verify. None if invalid."""
    w = np.asarray(w, dtype="float64").copy()
    if not np.all(np.isfinite(w)):
        return None
    w[np.abs(w - lower) < BOUND_SNAP] = lower
    w[np.abs(w - upper) < BOUND_SNAP] = upper
    total = w.sum()
    if abs(total - 1.0) > CHECK_TOL:
        return None
    # Renormalise only the weights strictly inside the bounds, so pinned weights stay put.
    free = (w > lower) & (w < upper)
    if free.any():
        w[free] += (1.0 - total) * w[free] / w[free].sum() if w[free].sum() > 0 else 0.0
    if np.any(w < lower - CHECK_TOL) or np.any(w > upper + CHECK_TOL):
        return None
    if abs(w.sum() - 1.0) > CHECK_TOL:
        return None
    return np.clip(w, lower, upper)


def solve(
    objective: Objective,
    gradient: Gradient,
    starts: Sequence[np.ndarray],
    lower: float,
    upper: float,
) -> SolveResult:
    """Minimise `objective` from each start; keep the lowest valid result."""
    n = len(starts[0])
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}
    ]
    bounds = [(lower, upper)] * n
    best: SolveResult | None = None
    last_message, last_iter = "no starting point", 0
    for x0 in starts:
        res = minimize(
            objective,
            np.asarray(x0, dtype="float64"),
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": FTOL, "maxiter": MAX_ITER},
        )
        last_message, last_iter = str(res.message), int(res.nit)
        if not res.success:
            continue
        w = clean_weights(res.x, lower, upper)
        if w is None:
            last_message = "solution failed the sum/bounds check"
            continue
        value = float(objective(w))
        if best is None or value < best.objective - 1e-15:
            best = SolveResult(w, True, str(res.message), int(res.nit), value)
    if best is None:
        return SolveResult(None, False, last_message, last_iter, float("nan"))
    return best
