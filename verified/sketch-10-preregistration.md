# Pre-registration — Sketch 10 ARE test (5-day bagging switching, AI ETFs)

**Written: 9 Nov 2026, 23:47 IST — BEFORE any data pull or test run.**
**Status: pre-registered. Gates below are frozen; they may not be changed
after the results are seen. If the test fails, the sketch stays a
hypothesis — no re-tuning, no second run with different gates.**

## Source
Paper 35: Perry Sadorsky, "Investing in AI Stocks: Machine-Learning
Forecasts, Portfolio Strategies, and Stress Tests", J. Risk Financ.
Manag. 2026, 19(11), 904. The paper reports, on a SINGLE 14-month
out-of-sample window (Mar 2025–Apr 2026), a 5-day bagging switching
portfolio with Sharpe 5.65 vs 0.02 equal-weight, max drawdown 3.0% vs
26.2%, after 100 bps trading costs — and the author himself says the
very high Sharpe "should be confirmed with longer samples". This test
is that longer-sample confirmation, run the ARE way.

## Hypothesis
A weekly bagging-classifier switching rule over the four AI sub-sector
ETFs (BUG / CLOU / BLOK / BOTZ) earns a positive risk-adjusted return
net of double costs over a multi-year walk-forward test, and survives
a Monte Carlo reshuffle.

## Universe and data
- ETFs: BUG (cybersecurity), CLOU (cloud), BLOK (blockchain),
  BOTZ (robotics/AI).
- Daily adjusted closes + volume from the Yahoo Finance chart API
  (query1.finance.yahoo.com/v8/finance/chart/<TICKER>), Nov 2019
  → 9 Nov 2026. Adjusted close (splits/dividends handled by Yahoo).
- No Stooq (bot-blocked). No other data source.

## Rule (exact)
- **Rebalance: weekly, at the Friday close** (last trading day of the
  week if Friday is a holiday). The new position applies from the
  next trading day.
- **Per ETF, one bagging classifier** (sklearn BaggingClassifier,
  DecisionTreeClassifier base, 100 estimators, `random_state=42`).
  It predicts the direction of the 5-day forward return.
- **Features (technical only, 10 per ETF, all computed from data up
  to the rebalance date — no lookahead):**
  1. 1-day return
  2. 5-day return
  3. 20-day return
  4. close / MA20
  5. close / MA50
  6. close / MA200
  7. RSI(14) (Wilder)
  8. MACD histogram / close, MACD(12,26,9)
  9. OBV 5-day % change (OBV from close and volume)
  10. 20-day volatility (std of daily returns)
- **Label:** 1 if the 5-day forward adjusted return > 0, else 0.
- **Decision:** hold the ETF if P(up) > 0.5, equal-weighted among
  the ETFs held; sit out the rest in cash. Cash earns 0%.
- **Training rows:** a row is usable only when its 5-day forward
  outcome is known by the training cutoff (no label lookahead).

## Walk-forward (expanding window)
- Folds are 3-month test blocks. The first test fold starts after at
  least 12 months of training data (training from Nov 2019, so the
  first fold starts ~Nov 2020). Each fold: train once on ALL data up
  to the fold start, test the next 3 months, roll forward.
- **The final 3-month fold is the holdout**: it is opened once and
  reported separately, as well as inside the overall stats.
- Early folds have a thin training set (the 200-day MA needs ~9.5
  months of history before the first feature-complete row exists).
  That is recorded, not fixed by moving the start date.

## Costs
- **200 bps per one-way switch** (double the paper's 100 bps — the
  ARE double-costs rule). Cost at a rebalance = 0.02 × one-way
  turnover, where one-way turnover = 0.5 × Σ|w_new − w_old| across
  the four ETFs plus the cash weight. Charged on the rebalance date.
- Cash earns 0%. No interest, no borrow, no leverage, no shorting.

## Pass gates — ALL must pass for "Verified"
1. Out-of-sample Sharpe (daily returns, rf = 0, annualised √252) ≥ 1.0
2. Max drawdown ≤ 15%
3. Worst walk-forward fold return ≥ −5%
4. Monte Carlo p < 0.05 — 1,000 block-bootstraps (5-day blocks) of
   the strategy's daily returns; observed Sharpe above the 95th
   percentile of the reshuffled Sharpes (p = share of reshuffled
   Sharpes ≥ observed Sharpe)
5. Positive total return net of 200 bps costs

## Benchmark (context only, NOT a gate)
Equal-weight buy-and-hold of the four ETFs over the same test
period, same 200 bps cost rule on its own rebalancing (weekly
rebalance to equal weights). Reported for comparison only.

## Reporting
Full results in `sketch-10-test-report.md`: per-fold table (return,
Sharpe, max DD), worst fold, overall stats, Monte Carlo p, holdout
fold separately, and a PASS/FAIL line per gate. Numbers exactly as
computed — no spin, no cherry-picking. If ANY gate fails, the
strategy is NOT verified, no deployment kit is written, and the
Verified section shows 0 verified with the failing gates named.

## What "verified" means here
A pass means: this rule, exactly as written above, survived a
multi-year walk-forward with double costs and a Monte Carlo
reshuffle. It is still a backtest. It is not a promise of future
returns, and it is not financial advice. Deployment is in the
user's OWN account only — prop firms ban third-party trading
(FTMO, The5ers, Topstep, Apex, FundedNext, E8; FundedNext
$25,000 penalty per breach), so no copier, no signals into
someone else's account, no managed prop trading.
