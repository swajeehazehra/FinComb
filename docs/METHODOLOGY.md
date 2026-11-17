# Methodology

How every number in FinComb is calculated, and which conventions it uses. When two
tools disagree on a Sharpe ratio, the reason is almost always a convention, so they
are all written down here.

## Conventions

| Convention | Choice | Where it lives |
|---|---|---|
| Price used | **Adjusted close** (splits and dividends included) | `services/market_data.py` |
| Return type | **Simple** returns for performance, log returns available for statistics | `analytics/returns.py` |
| Year length | **252** trading days | `analytics/constants.py` |
| Standard deviation | Sample (ddof = 1) | `analytics/risk.py` |
| Risk-free rate | Input as an **annual** rate, converted by compounding to a daily rate | `risk.periodic_rate` |
| Missing prices | Never filled. A gap gives a NaN return, not an invented one | `returns.simple_returns` |
| Several assets | Aligned on the dates **all** of them traded (inner join) by default | `market_data.load_prices` |
| Rolling windows | Trailing only, full window required (no partial or centred windows) | `risk.rolling_*` |

## Returns

Simple return on day *t*:

$$r_t = \frac{P_t}{P_{t-1}} - 1$$

Log return: $\ln(P_t / P_{t-1})$. Log returns add up over time; simple returns
compound: total return $= \prod_t (1 + r_t) - 1$.

**CAGR** (compound annual growth rate), with *n* daily returns:

$$\text{CAGR} = (1 + \text{total return})^{252 / n} - 1$$

So 252 returns are one year, and CAGR equals the total return over exactly one year.

**Resampling** to weekly, monthly or yearly compounds the daily returns inside
each period: $\prod (1 + r_{\text{daily}}) - 1$. Weeks end on Friday.

## Risk

**Annualised volatility:** $\sigma_{\text{daily}} \times \sqrt{252}$. Scaling by
$\sqrt{252}$ assumes daily returns are roughly independent; real returns show
some autocorrelation, so treat this as an approximation.

**Daily risk-free rate** from an annual rate $R_f$: $r_f = (1 + R_f)^{1/252} - 1$.
Compounding this back over 252 days gives exactly $R_f$.

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
and 0.

**Drawdown episodes** split the drawdown series into separate underwater spells.
Each one records the peak (the last day at the old high), the trough, the
recovery (the first day back at the old high, if it happened), the depth, and how
many trading days it took to fall and to recover.

**Calmar ratio:** CAGR / |maximum drawdown|.

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

## Known limitations

* **Survivorship bias:** only tickers you ingest are analysed, and today's tickers
  are by definition the survivors. Historical results for a universe chosen today
  look better than reality.
* **Adjusted prices are revised:** when a new dividend or split happens, the
  provider re-adjusts the whole history. Re-ingesting updates stored values.
* **Normality:** Sharpe and √252 scaling treat returns as roughly normal and
  independent. Fat tails and volatility clustering are not captured by these
  numbers.
