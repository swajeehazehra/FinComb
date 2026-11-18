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
| Several assets | Aligned on the dates **all** of them traded (inner join) by default, with a note under the results naming the symbol that limits the start or end of the period, and `from`/`to` columns showing each symbol's exact window; `--align outer` keeps each symbol's own history | `market_data.load_prices` |
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

## Portfolio construction

All four methods are **long-only and fully invested**: weights sum to 1 and each
lies in `[min_weight, max_weight]` (defaults 0 and 1). The bounds are checked before
anything is solved: N assets need N × max_weight ≥ 1 and N × min_weight ≤ 1.

**Estimation window.** By default the last 252 daily returns (`lookback`) ending at
`end` (default: the latest date every asset has); with an explicit `start` the
whole `[start, end]` range is used. Assets are inner-aligned. The window needs at
least max(60, 2N) returns, because a covariance matrix estimated from barely more
rows than assets is singular or very noisy. Daily means μ and covariances Σ come
from this window and are annualised with 252.

**Equal weight**: 1/N each. No estimation error at all, which is why it is hard to
beat out of sample; it ignores risk differences, though.

**Inverse volatility**: $w_i \propto 1/\sigma_i$. Equal risk contributions when
all correlations are equal ("naive risk parity"); it ignores correlations. With
bounds, $w_i = \text{clip}(c/\sigma_i, \text{min}, \text{max})$ with the single
scalar $c$ that makes them sum to 1. The sum is continuous and non-decreasing in
$c$, from N × min_weight to N × max_weight, so a solution exists whenever the
bounds are feasible; assets not pinned at a bound keep their 1/σ proportions. An
asset with zero volatility in the window is rejected.

**Minimum variance**: minimise $w^\top \Sigma w$. It needs no expected returns,
and the problem is convex, so the optimum is global. Unconstrained,
$w^* = \Sigma^{-1}\mathbf{1} / \mathbf{1}^\top\Sigma^{-1}\mathbf{1}$; at a bounded
optimum every asset strictly inside its bounds has the same marginal variance
$(\Sigma w)_i$. It tends to concentrate in the least volatile assets, and it is
minimum variance *in sample*: realised variance afterwards is higher.

**Maximum Sharpe**: maximise $(w^\top\mu - r_f)/\sqrt{w^\top\Sigma w}$, scaled by
√252 so it means the same as the Sharpe ratio elsewhere. Unconstrained,
$w \propto \Sigma^{-1}(\mu - r_f\mathbf{1})$. The ratio is quasi-concave where the
excess return is positive, so a local maximum is the global one. **Mean returns
from a year of data are mostly noise** (their standard error is about σ/√years,
so ±25% a year for a 25%-volatility stock), and the optimiser treats that noise as
signal: expect concentrated, unstable weights unless `max_weight` caps them.

**When no asset beats the risk-free rate.** If every asset's estimated excess
return is ≤ 0, every long-only portfolio has a negative expected Sharpe ratio.
Maximising a negative ratio rewards a *larger* denominator, i.e. **higher
volatility**, so the "optimal" weights can look perverse (for example 100% in the
most volatile asset). FinComb still returns them, because they are the correct
answer to the question asked, but it never switches to another method silently:
the result has `non_positive_excess_return: true` and a warning, shown above the
weights in the CLI and in `warnings` in the API, and kept with a saved portfolio.
**Such a portfolio is not an attractive investment and should not be read as a
recommendation**; use minimum variance, another window or a lower risk-free rate.
Because the problem is no longer well-behaved, SLSQP is started from equal
weights, inverse volatility and one corner per asset, and the best answer kept.

## The optimiser

Minimum variance and maximum Sharpe use SciPy's SLSQP with analytic gradients
(2Σw, and the quotient rule for the Sharpe ratio), the equality constraint
sum(w) = 1, box bounds, `ftol = 1e-12`, at most 500 iterations, starting from
equal weights. Σ is annualised first so the objective isn't of order 1e-4, where
an absolute tolerance would stop too early.

**Covariance regularisation.** The default estimator is **Ledoit-Wolf
shrinkage** toward a scaled identity, $\hat\Sigma = \delta F + (1-\delta) S$ with
$F = \operatorname{tr}(S)/N \cdot I$. The intensity δ is the Ledoit-Wolf (2004)
formula that minimises expected estimation error; it is reported with every
result (small for 6 assets and 252 days, large for 30 assets and 60 days).
`covariance=sample` uses $S$ alone. In both cases a matrix whose smallest
eigenvalue is not above $10^{-12}$ times its largest (duplicate, collinear or
flat assets) gets the smallest ridge $\varepsilon \cdot \operatorname{tr}(\Sigma)/N
\cdot I$, ε = 1e-10 … 1e-4, that fixes it, with a warning.

**Checking the answer.** SciPy's `success` flag is not trusted alone: weights
within 1e-10 of a bound are snapped to it, and the result must sum to 1 and lie
in the bounds within 1e-8. It must also be at least as good as every simple
feasible portfolio (equal weight, inverse volatility, and for maximum Sharpe one
corner per asset).

**Fallback.** If the covariance can't be repaired, SLSQP fails, or the answer
fails a check, the weights are bounded inverse volatility instead.
`fallback_used`, `fallback_reason`, `effective_method: inverse_vol` and a warning
say so, and a saved portfolio stores both the requested and the effective method.

The statistics reported with constructed weights (return, volatility, Sharpe)
are **in-sample**: they describe the window the weights were fitted on, which
flatters them. Evaluate a portfolio on a later period with the risk report.

## Portfolio risk

**Portfolio returns** are *constant mix*: $r_{p,t} = \sum_i w_i r_{i,t}$, as if
rebalanced to the weights every day. That is the standard ex-ante convention and
keeps the identities exact (std of $r_p$ = $\sqrt{w^\top S w}$). Drift between
rebalances belongs to the backtester (Milestone 5). Total return, CAGR, Sharpe,
drawdown and the rest come from the existing performance code applied to the
portfolio's value series.

**VaR and CVaR** are **1-day** losses reported as **positive numbers**, at
confidence α (default 95%). Historical VaR is minus the $(1-\alpha)$ quantile of
the portfolio's daily returns, using **linear interpolation** between sorted
returns (NumPy's default, Excel's PERCENTILE.INC). Historical CVaR (expected
shortfall) is the average of that **same** quantile function over the tail:

$$\text{CVaR}_\alpha = -\frac{1}{1-\alpha}\int_0^{1-\alpha} Q(u)\,du$$

$Q$ is piecewise linear between the sorted returns, so the integral is computed
exactly with trapezoids. A fractional tail (12.6 observations for 252 days at
95%) needs no rounding, and because $Q$ never decreases its average below
$1-\alpha$ is at most $Q(1-\alpha)$: **CVaR ≥ VaR always**. Gaussian VaR,
$-(\bar r + z_{1-\alpha}\sigma)$, and CVaR, $\sigma\varphi(z_{1-\alpha})/(1-\alpha) -
\bar r$, are shown alongside; a historical figure well above the Gaussian one
means fatter tails than normal. With fewer than 10 observations in the tail (for
example 99% on one year of data) the report warns that the historical figures are
unreliable. No √h scaling to longer horizons is applied.

**Beta and alpha** against a benchmark (default SPY): $\beta = \text{Cov}(r_p,
r_b)/\text{Var}(r_b)$, the OLS slope, on the dates both have. Jensen's alpha
$= \overline{r_p - r_f} - \beta\,\overline{r_b - r_f}$, annualised ×252. With a
constant risk-free rate, beta is the same for raw and excess returns. Correlation
and R² are reported too. A zero-variance benchmark gives null beta.

**Risk contributions** (annualised, Euler decomposition): marginal
$\text{MCR}_i = (\Sigma w)_i/\sigma_p$, component $\text{CR}_i = w_i\,\text{MCR}_i$
(they sum exactly to $\sigma_p$), and percentage $\text{CR}_i/\sigma_p$ (sums to
100%). A negative contribution means the asset lowers total risk. The report uses
the plain sample covariance of the analysis period, since it describes what
happened; shrinkage only helps the optimiser.

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
  numbers. Gaussian VaR/CVaR share this assumption; the historical figures don't.
* **Estimation error in optimised portfolios:** minimum-variance and maximum-Sharpe
  weights are fitted to one window of history. Their in-sample statistics are
  optimistic, and maximum-Sharpe weights in particular change a lot from one
  window to the next.
