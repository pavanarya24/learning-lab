# Test report — Sketch 10 ARE test (5-day bagging switching, AI ETFs)

**Run: 09 Nov 2026, 23:55 IST. Pre-registration: `verified/sketch-10-preregistration.md`, written 9 Nov 2026 23:47 IST — BEFORE this run. Gates frozen; no re-tuning.**

## Verdict

**FAIL — NOT VERIFIED** — one or more gates fail. Sketch 10 stays a hypothesis. No deployment kit.

## Overall (out-of-sample, test period only)

- Test period: 2020-11-02 → 2026-11-09 (1512 trading days, 24 walk-forward folds of 3 months)
- Total return net of 200 bps costs: **-86.94%**
- Annualised Sharpe (rf = 0): **-1.01**
- Max drawdown: **-88.99%**
- Total costs paid: 279.50% of capital
- Gross return BEFORE costs: +118.48% (Sharpe 0.60) — so costs, not the raw signal alone, do most of the damage: the technical-only signal flips holdings almost every week and the 200 bps switch charge compounds against it
- Weekly rebalances in test period: 315; weeks fully in cash: 16 (5% of weeks)
- Classifier direction accuracy (weekly, 5-day outcome): 51.4% (1252 predictions)
- Training rows at first fold: BUG=49, CLOU=49, BLOK=49, BOTZ=49 (thin — the 200-day MA needs ~9.5 months of history; recorded, not fixed)

## Per-fold results

| Fold | Start | End | Days | Return | Sharpe | Max DD |
|---:|---|---|---:|---:|---:|---:|
| 1 | 2020-11-02 | 2021-01-29 | 61 | +3.28% | 0.69 | -10.34% |
| 2 | 2021-02-01 | 2021-04-30 | 63 | -1.62% | -0.10 | -16.40% |
| 3 | 2021-05-03 | 2021-07-30 | 63 | -3.78% | -0.49 | -11.32% |
| 4 | 2021-08-02 | 2021-10-29 | 64 | +7.02% | 1.37 | -13.03% |
| 5 | 2021-11-01 | 2022-01-31 | 63 | -26.68% | -3.68 | -33.83% |
| 6 | 2022-02-01 | 2022-04-29 | 62 | -22.87% | -2.38 | -24.47% |
| 7 | 2022-05-02 | 2022-07-29 | 62 | -16.93% | -1.44 | -26.61% |
| 8 | 2022-08-01 | 2022-10-31 | 65 | -20.00% | -2.61 | -26.32% |
| 9 | 2022-11-01 | 2023-01-31 | 62 | -0.49% | 0.11 | -16.32% |
| 10 | 2023-02-01 | 2023-04-28 | 61 | -12.94% | -1.59 | -21.33% |
| 11 | 2023-05-01 | 2023-07-31 | 63 | +11.45% | 1.82 | -6.20% |
| 12 | 2023-08-01 | 2023-10-31 | 65 | -25.27% | -4.70 | -26.35% |
| 13 | 2023-11-01 | 2024-01-31 | 62 | +20.56% | 3.70 | -6.86% |
| 14 | 2024-02-01 | 2024-04-30 | 62 | -8.95% | -1.38 | -18.52% |
| 15 | 2024-05-01 | 2024-07-31 | 63 | -0.98% | -0.08 | -9.08% |
| 16 | 2024-08-01 | 2024-10-31 | 65 | -9.06% | -1.24 | -13.54% |
| 17 | 2024-11-01 | 2025-01-31 | 61 | -8.86% | -1.56 | -15.13% |
| 18 | 2025-02-03 | 2025-04-30 | 61 | -21.40% | -2.22 | -31.54% |
| 19 | 2025-05-01 | 2025-07-31 | 63 | +1.13% | 0.31 | -8.50% |
| 20 | 2025-08-01 | 2025-10-31 | 65 | +2.86% | 0.59 | -5.93% |
| 21 | 2025-11-03 | 2026-01-30 | 61 | -17.89% | -3.67 | -18.35% |
| 22 | 2026-02-02 | 2026-04-30 | 62 | -14.13% | -1.81 | -20.19% |
| 23 | 2026-05-01 | 2026-07-31 | 63 | -11.30% | -1.29 | -23.51% |
| 24 | 2026-08-03 | 2026-11-09 | 70 | -0.05% | 0.15 | -12.25% | ← HOLDOUT

- **Worst fold:** fold 5 (2021-11-01 → 2022-01-31), return -26.68%, Sharpe -3.68, max DD -33.83%

## Holdout fold (final 3 months, opened once, reported separately)

- Fold 24: 2026-08-03 → 2026-11-09, return -0.05%, Sharpe 0.15, max DD -12.25%

## Monte Carlo (1,000 block-bootstraps, 5-day blocks, seed 42)

- Observed Sharpe: -1.01
- 95th percentile of reshuffled Sharpes: -0.39
- **p = 0.487** (share of reshuffled Sharpes ≥ observed)

## Gates

| Gate | Value | Threshold | Result |
|---|---|---|---|
| 1. Out-of-sample Sharpe >= 1.0 | -1.01 | >= 1.0 | **FAIL** |
| 2. Max drawdown <= 15% | -88.99% | <= -15% max | **FAIL** |
| 3. Worst walk-forward fold return >= -5% | -26.68% | >= -5% | **FAIL** |
| 4. Monte Carlo p < 0.05 (1,000 x 5-day block bootstraps) | 0.487 | < 0.05 | **FAIL** |
| 5. Positive total return net of 200 bps costs | -86.94% | > 0 | **FAIL** |

## Benchmark (context only, NOT a gate)

- Equal-weight buy-and-hold, same period, same 200 bps cost rule: total return +64.15%, Sharpe 0.43, max drawdown -54.93%

## What this means

The rule did NOT survive the pre-registered gates. The most likely reading: the paper's 5.65 Sharpe was a single 14-month window (Mar 2025–Apr 2026) — exactly the limitation its author flagged. On a longer multi-year sample at double costs, the edge does not clear the bar. Note the failure is not only a costs story: even BEFORE costs the gross Sharpe is 0.60, below the 1.0 gate, and the classifier is right only 51.4% of the time — near chance. The technical-only variant (no macro features, unlike the paper) flips holdings almost weekly, and 200 bps a switch compounds that churn into a −87% net result. The sketch stays a hypothesis; it is frozen, not re-tuned. The next candidates in the queue are Sketch 6 (news-echo ORB) and Sketch 8 (BOCPD regime gate).

## Reproducibility

- Data: `verified/data/{BUG,CLOU,BLOK,BOTZ}.csv` — Yahoo Finance chart API, adjusted closes, 1 Nov 2019 → 9 Nov 2026 (1,764 rows each).
- Script: `verified/run_sketch10_test.py` (BaggingClassifier, DecisionTreeClassifier base, 100 estimators, random_state=42; Monte Carlo seed 42).
- Costs: 200 bps per one-way switch, charged on the rebalance date; cash earns 0%; no leverage, no shorting.
- Implementation note: the first run charged a phantom full-entry cost at every fold boundary (position reset to cash per fold). That was a bug, not a gate change — the pre-registered rule charges costs on ACTUAL switches, so the position now carries across folds and this report is from the corrected run.
