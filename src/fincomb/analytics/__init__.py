"""Pure quantitative analytics: pandas in, pandas/floats out. No I/O, no database.

Conventions (see docs/METHODOLOGY.md):
* inputs are adjusted close prices or simple daily returns, indexed by date
* 252 trading days per year
* functions accept a Series (one asset) or a DataFrame (one column per asset)
"""

from fincomb.analytics.constants import TRADING_DAYS

__all__ = ["TRADING_DAYS"]
