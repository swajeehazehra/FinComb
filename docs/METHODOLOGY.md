# Methodology

How every number in FinComb is calculated, and which conventions it uses. When two
tools disagree on a Sharpe ratio, the reason is almost always a convention, so they
are all written down here.

## Conventions

| Convention | Choice | Where it lives |
|---|---|---|
| Price used | **Adjusted close** (splits and dividends included) | `services/market_data.py` |
| Return type | **Simple** returns for performance, log returns available for statistics | `analytics/returns.py` |
| Year length for volatility, Sharpe, Sortino | **252** trading days | `analytics/constants.py` |
| Year length for CAGR | **365.25** calendar days between first and last price | `analytics/constants.py` |
| Standard deviation | Sample (ddof = 1) | `analytics/risk.py` |
| Risk-free rate | Input as an **annual** rate, converted by compounding to a daily rate | `risk.periodic_rate` |
| Missing prices | Never filled. A gap gives a NaN return, not an invented one | `returns.simple_returns` |
| Missing returns when compounding | **Error by default** (`missing="raise"`); `missing="zero"` treats a gap as a flat day only when asked | `returns.handle_missing` |
| Several assets | Aligned on the dates **all** of them traded (inner join) by default, with a warning naming the symbol that limits the period; `--align outer` keeps each symbol's own history | `market_data.load_prices` |
| Resampled periods | Labelled with the **last actual trading day** in the period | `returns.resample_returns` |
| Rolling windows | Trailing only, full window required (no partial or centred windows) | `risk.rolling_*` |

## Returns

Simple return on day *t*:

$$r_t = \frac{P_t}{P_{t-1}} - 1$$

Log return: $\ln(P_t / P_{t-1})$. Log returns add up over time; simple returns
compound: total return $= \prod_t (1 + r_t) - 1$.

**CAGR** (compound annual growth rate) uses **elapsed calendar time** from the
date of the first price (the day the money is invested) to the date of the last
price:

$$\text{CAGR} = (1 + \text{total return})^{1 / Y} - 1, \qquad Y = \frac{\text{days between first and last price}}{365.25}$$

Calendar time is used because it doesn't depend on how many rows the data has:
a missing day, or a date dropped when aligning several symbols, leaves CAGR
unchanged. (Counting returns and dividing by 252 would overstate CAGR in those
cases.) When only returns are available, the start is assumed to be one business
day before the first return.

**Resampling** to weekly, monthly or yearly compounds the daily returns inside
each period: $\prod (1 + r_{\text{daily}}) - 1$. Each period is labelled with its
last actual trading day (for example 2024-08-30, not Saturday 2024-08-31), so it
joins cleanly back onto daily data.

**Missing returns.** Ratio statistics (volatility, Sharpe, Sortino, correlation)
skip missing observations. Compounding (wealth index, total return, CAGR,
drawdowns) **refuses to run over a gap** and raises an error naming the asset and
dates, because silently treating a gap as a 0 % day can hide data problems.
Pass `missing="zero"` to treat gaps as flat days on purpose. NaNs before an
asset's first return or after its last one are not gaps (the asset just wasn't
trading yet) and are left alone. Per-symbol reports use each symbol's own price
history, so a missing day just means the next return covers two days.

## Risk

**Annualised volatility:** $\sigma_{\text{daily}} \times \sqrt{252}$. Scaling by
$\sqrt{252}$ assumes daily returns are roughly independent; real returns show
some autocorrelation, so treat this as an approximation.

**Daily risk-free rate** from an annual rate $R_f$: $r_f = (1 + R_f)^{1/252} - 1$.
Compounding this back over 252 days gives exactly $R_f$. The same daily rate is
used by Sharpe, Sortino and rolling Sharpe.

`--risk-free` is a **constant** for the whole period. Real short-term rates
change a lot: 3-month US Treasury bills were near 0 % in 2015–16 and 2020–21 and
above 5 % in 2023–24, averaging roughly 2 % over 2015–2026. A constant 4 % over
that span sets the hurdle too high and understates every Sharpe and Sortino.
Pick a rate that matches the period (about 0.02 for 2015–2026), or 0 to compare
raw risk-adjusted returns. A time-varying risk-free series is planned for the
portfolio and backtesting milestones.

**Sharpe ratio:**

$$\text{Sharpe} = \frac{\overline{r - r_f}}{\sigma(r - r_f)} \times \sqrt{252}$$

When volatility is zero (a constant series) the ratio is undefined and returned as
NaN, never as infinity.

**Sortino ratio** only penalises returns below the target (the risk-free rate):

$$\text{DD} = \sqrt{\text{mean}\big(\min(r - r_f, 0)^2\big)} \times \sqrt{252},
\qquad \text{Sortino} = \frac{\overline{r - r_f} \times 252}{\text{DD}}$$

The mean is taken over **all** days, not only losing days (the standard
definition). A series with no losing days has an undefined Sortino (NaN).

**Drawdown** at each date is the percentage below the highest value reached so far:

$$\text{DD}_t = \frac{W_t}{\max(1, \max_{s \le t} W_s)} - 1, \qquad W_t = \prod_{s \le t}(1 + r_s)$$

The starting capital (1) counts as the first peak, so a loss on day one shows up
as a drawdown. **Maximum drawdown** is the minimum of this series, between −100 %
and 0. Drawdowns smaller than $10^{-10}$ are treated as 0, because compounding
rounding can leave a fully recovered value a hair below its old peak (for
example −1e−16), which would otherwise hide the recovery.

**Drawdown episodes** split the drawdown series into separate underwater spells.
Each one records the peak (the last day at the old high), the trough, the
recovery (the first day back at the old high, if it happened), the depth, and how
many trading days it took to fall and to recover.

**Calmar ratio:** CAGR / |maximum drawdown|, both measured over the **whole
period analysed**. The original definition uses only the last 36 months; to get
that, pass a 3-year window with `--start`. A negative CAGR gives a negative
Calmar, and no drawdown at all gives n/a.

## Correlation

Correlation is always computed on **returns**, never on prices. Two trending price
series look highly correlated even when their daily moves are unrelated.

* **Pearson** measures linear co-movement.
* **Spearman** uses ranks, so it is robust to outliers and captures any monotonic
  relationship.

Each pair needs at least 20 overlapping returns, otherwise the cell is NaN.

## Bias and reproducibility controls

* Rolling statistics use only past data. A test changes every return after a
  cutoff date and asserts that no rolling value before the cutoff changes.
* Analytics are pure functions with no randomness or I/O, so the same prices
  always give the same numbers.
* Results are cross-checked against an independent NumPy implementation, and
  property-based tests (Hypothesis) check invariants on hundreds of random series,
  for example "total return = last price / first price − 1" and "scaling all prices
  by a constant changes no statistic".

## Adjusted prices and look-ahead: the rule for backtesting

Adjusted prices are recalculated backwards every time a dividend or split
happens, so an adjusted price from 2016 already "knows" about dividends paid
later. That is harmless for **returns and ratios**, which come out the same, but
using adjusted price **levels** in a trading rule leaks future information
(for example "buy when the price is below $100", or a fixed-dollar stop).

The backtester (Milestone 5) will therefore follow one rule:

* **Signals** are built only from returns and ratios of `adj_close` (momentum,
  z-scores, moving-average ratios), never from adjusted price levels or dollar
  thresholds.
* **Trades fill** at the raw `open`/`close` actually traded that day, with costs
  and slippage applied to that raw price.

## Known limitations

* **Survivorship bias:** only tickers you ingest are analysed, and today's tickers
  are by definition the survivors. Historical results for a universe chosen today
  look better than reality.
* **Adjusted prices are revised:** when a new dividend or split happens, the
  provider re-adjusts the whole history. Re-ingesting updates stored values.
* **Normality:** Sharpe and √252 scaling treat returns as roughly normal and
  independent. Fat tails and volatility clustering are not captured by these
  numbers.
