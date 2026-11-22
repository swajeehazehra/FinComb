"""What makes a backtest reproducible: the code version and a hash of its input data."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

CODE_VERSION_ENV = "FINCOMB_CODE_VERSION"


def data_hash(frames: dict[str, pd.DataFrame | pd.Series]) -> str:
    """SHA-256 over the exact input prices: names, symbols, dates and float64 values.

    Frames and columns are sorted by name, so the same data in another column order
    gives the same hash; any change to any value (or date) changes it.
    """
    h = hashlib.sha256()
    for name in sorted(frames):
        frame = frames[name]
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(frame.name or name)
        frame = frame[sorted(map(str, frame.columns))] if len(frame.columns) else frame
        h.update(f"[{name}]".encode())
        h.update("\x1f".join(map(str, frame.columns)).encode())
        dates = pd.DatetimeIndex(frame.index).as_unit("ns").asi8
        h.update(np.ascontiguousarray(dates, dtype="<i8").tobytes())
        h.update(np.ascontiguousarray(frame.to_numpy(dtype="float64"), dtype="<f8").tobytes())
    return h.hexdigest()


def code_version() -> str:
    """Git commit of the running code: FINCOMB_CODE_VERSION if set, else `git rev-parse`."""
    env = os.environ.get(CODE_VERSION_ENV, "").strip()
    if env:
        return env[:40]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        sha = out.stdout.strip()
        return sha[:40] if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
