#!/usr/bin/env python3
"""Sketch 10 ARE test — 5-day bagging switching on BUG/CLOU/BLOK/BOTZ.
Implements sketch-10-preregistration.md EXACTLY. No gate may be changed.
Run: /tmp/are-venv/bin/python run_sketch10_test.py
Outputs: sketch-10-test-report.md + console summary.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = "/home/hatch/workspace/research/learning-lab/verified/data"
OUT = "/home/hatch/workspace/research/learning-lab/verified/sketch-10-test-report.md"
TICKERS = ["BUG", "CLOU", "BLOK", "BOTZ"]
COST_BPS = 0.02          # 200 bps per one-way switch (double the paper's 100 bps)
RANDOM_STATE = 42
N_FOLDS_HOLDOUT_LAST = True

# ---------------------------------------------------------------- features
def add_features(df):
    df = df.copy()
    c = df["adjclose"]
    v = df["volume"].fillna(0)
    ret = c.pct_change()
    df["ret_1d"] = ret
    df["ret_5d"] = c.pct_change(5)
    df["ret_20d"] = c.pct_change(20)
    df["c_ma20"] = c / c.rolling(20).mean()
    df["c_ma50"] = c / c.rolling(50).mean()
    df["c_ma200"] = c / c.rolling(200).mean()
    # RSI(14), Wilder
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    # MACD(12,26,9) histogram / close
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - sig) / c
    # OBV 5-day % change
    direction = np.sign(delta.fillna(0))
    obv = (direction * v).cumsum()
    df["obv_5d"] = obv.pct_change(5)
    # 20-day volatility (std of daily returns)
    df["vol20"] = ret.rolling(20).std()
    # label: 5-day forward return > 0
    df["fwd5"] = c.shift(-5) / c - 1
    df["label"] = (df["fwd5"] > 0).astype(int)
    return df

FEATURES = ["ret_1d", "ret_5d", "ret_20d", "c_ma20", "c_ma50", "c_ma200",
            "rsi14", "macd_hist", "obv_5d", "vol20"]

# ---------------------------------------------------------------- data
frames = {}
for t in TICKERS:
    df = pd.read_csv(f"{DATA_DIR}/{t}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    frames[t] = add_features(df)
dates = frames[TICKERS[0]]["date"]  # all four share the same trading calendar
n = len(dates)
assert all(len(frames[t]) == n for t in TICKERS), "calendars differ"

# ---------------------------------------------------------------- folds
# Quarterly fold starts: Nov 2020, Feb 2021, May 2021, Aug 2021, ... Aug 2026.
fold_starts = []
for year in range(2020, 2027):
    for m in (11, 2, 5, 8):
        if year == 2020 and m != 11:
            continue
        if year == 2026 and m > 8:
            continue
        d = pd.Timestamp(year=year, month=m, day=1)
        # first trading day on/after the 1st
        idx = dates.searchsorted(d)
        if idx < n:
            fold_starts.append(idx)
fold_starts = sorted(set(fold_starts))
fold_ends = fold_starts[1:] + [n]
print(f"folds: {len(fold_starts)} | first fold starts {dates.iloc[fold_starts[0]].date()} "
      f"| last fold starts {dates.iloc[fold_starts[-1]].date()} | data ends {dates.iloc[-1].date()}")

# Rebalance dates: last trading day of each week (Friday close; holiday -> last day)
df0 = frames[TICKERS[0]]
week_id = df0["date"].dt.to_period("W-FRI")
reb_idx = df0.groupby(week_id, sort=True).tail(1).index.tolist()

# ---------------------------------------------------------------- walk-forward
# position weights per day (ETF weights, 4 cols) + daily gross returns
weights = np.zeros((n, 4))
daily_ret = np.zeros(n)          # strategy daily return, net of costs
costs_paid = np.zeros(n)
pred_log = []                    # (date, ticker, p_up, held, fwd5 realised)
fold_rows = []

# prev_w persists ACROSS folds: the equity curve is continuous, so the
# cost at a fold's first rebalance is vs the previous fold's last
# position, not vs cash. (Bug in v1: prev_w reset per fold, charging a
# phantom full-entry cost at every fold boundary. Fixed before the
# reported run; the pre-registered cost rule is unchanged.)
prev_w = np.zeros(4)
for fi, (fs, fe) in enumerate(zip(fold_starts, fold_ends)):
    # --- train: rows with features complete and label known by fold start
    # label of row i is known at i+5; require i+5 <= fs (i.e. i <= fs-5)
    train_end = fs - 5
    models = {}
    train_n = {}
    for j, t in enumerate(TICKERS):
        df = frames[t]
        tr = df.iloc[:train_end + 1].dropna(subset=FEATURES + ["label"])
        train_n[t] = len(tr)
        m = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                              n_estimators=100, random_state=RANDOM_STATE)
        m.fit(tr[FEATURES].values, tr["label"].values)
        models[t] = m
    # --- test: weekly rebalances inside [fs, fe)
    fold_rebs = [r for r in reb_idx if fs <= r < fe]
    for r in fold_rebs:
        probs = []
        for j, t in enumerate(TICKERS):
            df = frames[t]
            row = df.iloc[r][FEATURES]
            if row.isna().any():
                probs.append(0.5)  # no signal -> sit out (P=0.5 is not > 0.5)
            else:
                probs.append(models[t].predict_proba(row.values.reshape(1, -1))[0][1])
        held = np.array([1.0 if p > 0.5 else 0.0 for p in probs])
        k = held.sum()
        w = held / k if k > 0 else np.zeros(4)
        # cost: 200 bps x one-way turnover (incl. cash weight)
        prev_full = np.append(prev_w, 1 - prev_w.sum())
        new_full = np.append(w, 1 - w.sum())
        turnover = 0.5 * np.abs(new_full - prev_full).sum()
        cost = COST_BPS * turnover
        costs_paid[r] += cost
        # weights apply from the next trading day
        if r + 1 < n:
            weights[r + 1:, :] = w
        prev_w = w
        # log predictions + realised 5-day outcome (diagnostic only)
        for j, t in enumerate(TICKERS):
            fwd = frames[t].iloc[r]["fwd5"]
            pred_log.append((dates.iloc[r].date(), t, round(float(probs[j]), 3),
                             bool(held[j]), None if pd.isna(fwd) else round(float(fwd), 4)))
    # fold stats are computed after the full run from the equity curve

# ---------------------------------------------------------------- equity curve
# daily gross return of the held ETFs
ret_mat = np.zeros((n, 4))
for j, t in enumerate(TICKERS):
    ret_mat[:, j] = frames[t]["adjclose"].pct_change().fillna(0).values
gross = (weights * ret_mat).sum(axis=1)
daily_ret = gross - costs_paid
# mask to test period only (first fold start onwards)
test_start = fold_starts[0]
test_ret = daily_ret[test_start:]
# diagnostic: gross (pre-cost) equity, so the report can show how much of
# the result is signal vs churn. Not a gate.
gross_test = gross[test_start:]
gross_equity = np.cumprod(1 + gross_test)
gross_total = float(gross_equity[-1] - 1)
gross_sharpe = float((gross_test.mean() / gross_test.std(ddof=1)) * np.sqrt(252)) if gross_test.std(ddof=1) > 0 else 0.0
equity = np.cumprod(1 + test_ret)
total_return = equity[-1] - 1
mean_d, std_d = test_ret.mean(), test_ret.std(ddof=1)
sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0.0
peak = np.maximum.accumulate(equity)
max_dd = float(np.min(equity / peak - 1))
n_days = len(test_ret)
n_weeks = len([r for r in reb_idx if r >= test_start])
total_cost = float(costs_paid[test_start:].sum())

# per-fold stats (segments of the same equity curve)
fold_stats = []
for fi, (fs, fe) in enumerate(zip(fold_starts, fold_ends)):
    seg = daily_ret[fs:fe]
    eq = np.cumprod(1 + seg)
    f_ret = eq[-1] - 1
    f_sh = (seg.mean() / seg.std(ddof=1)) * np.sqrt(252) if seg.std(ddof=1) > 0 and len(seg) > 1 else 0.0
    pk = np.maximum.accumulate(eq)
    f_dd = float(np.min(eq / pk - 1))
    fold_stats.append({
        "fold": fi + 1,
        "start": str(dates.iloc[fs].date()), "end": str(dates.iloc[fe - 1].date()),
        "days": len(seg), "return": float(f_ret), "sharpe": float(f_sh),
        "max_dd": f_dd, "holdout": fi == len(fold_starts) - 1,
    })
worst = min(fold_stats, key=lambda f: f["return"])
holdout = fold_stats[-1]

# ---------------------------------------------------------------- benchmark (context only)
# Equal-weight buy-and-hold, weekly rebalance to 0.25 each, same cost rule
bw = np.zeros((n, 4))
bcost = np.zeros(n)
prev = np.zeros(4)
for r in reb_idx:
    if r < test_start:
        continue
    w = np.array([0.25, 0.25, 0.25, 0.25])
    prev_full = np.append(prev, 1 - prev.sum())
    new_full = np.append(w, 1 - w.sum())
    bcost[r] += COST_BPS * 0.5 * np.abs(new_full - prev_full).sum()
    if r + 1 < n:
        bw[r + 1:, :] = w
    prev = w
bret = (bw * ret_mat).sum(axis=1) - bcost
btest = bret[test_start:]
beq = np.cumprod(1 + btest)
b_sharpe = (btest.mean() / btest.std(ddof=1)) * np.sqrt(252)
bpk = np.maximum.accumulate(beq)
b_dd = float(np.min(beq / bpk - 1))
b_total = float(beq[-1] - 1)

# ---------------------------------------------------------------- Monte Carlo
rng = np.random.default_rng(RANDOM_STATE)
BLOCK = 5
n_blocks = int(np.ceil(n_days / BLOCK))
mc_sharpes = []
for _ in range(1000):
    idx = rng.integers(0, n_days - BLOCK + 1, size=n_blocks)
    sample = np.concatenate([test_ret[i:i + BLOCK] for i in idx])[:n_days]
    s = sample.std(ddof=1)
    mc_sharpes.append((sample.mean() / s) * np.sqrt(252) if s > 0 else 0.0)
mc_sharpes = np.array(mc_sharpes)
p_value = float((mc_sharpes >= sharpe).mean())
p95 = float(np.percentile(mc_sharpes, 95))

# ---------------------------------------------------------------- diagnostics
pred_df = pd.DataFrame(pred_log, columns=["date", "ticker", "p_up", "held", "fwd5"])
pred_df = pred_df.dropna(subset=["fwd5"])
pred_df["correct"] = ((pred_df["p_up"] > 0.5) == (pred_df["fwd5"] > 0))
accuracy = float(pred_df["correct"].mean())
weeks_in = int(pred_df.groupby("date")["held"].sum().gt(0).sum())
weeks_total = int(pred_df["date"].nunique())
cash_weeks = weeks_total - weeks_in

# ---------------------------------------------------------------- gates
gates = [
    ("1. Out-of-sample Sharpe >= 1.0", sharpe, ">= 1.0", sharpe >= 1.0),
    ("2. Max drawdown <= 15%", max_dd, "<= -15% max", max_dd >= -0.15),
    ("3. Worst walk-forward fold return >= -5%", worst["return"], ">= -5%", worst["return"] >= -0.05),
    ("4. Monte Carlo p < 0.05 (1,000 x 5-day block bootstraps)", p_value, "< 0.05", p_value < 0.05),
    ("5. Positive total return net of 200 bps costs", total_return, "> 0", total_return > 0),
]
passed = all(g[3] for g in gates)
verdict = "PASS — VERIFIED" if passed else "FAIL — NOT VERIFIED"

# ---------------------------------------------------------------- report
lines = []
lines.append("# Test report — Sketch 10 ARE test (5-day bagging switching, AI ETFs)\n")
import datetime as _dt
_run_ts = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %H:%M IST")
lines.append(f"**Run: {_run_ts}. Pre-registration: "
             f"`verified/sketch-10-preregistration.md`, written 9 Nov 2026 23:47 IST — "
             f"BEFORE this run. Gates frozen; no re-tuning.**\n")
lines.append("## Verdict\n")
lines.append(f"**{verdict}** — {'all 5 gates pass' if passed else 'one or more gates fail'}. "
             f"{'Sketch 10 becomes Verified Strategy #1.' if passed else 'Sketch 10 stays a hypothesis. No deployment kit.'}\n")
lines.append("## Overall (out-of-sample, test period only)\n")
lines.append(f"- Test period: {dates.iloc[test_start].date()} → {dates.iloc[-1].date()} "
             f"({n_days} trading days, {len(fold_stats)} walk-forward folds of 3 months)")
lines.append(f"- Total return net of 200 bps costs: **{total_return:+.2%}**")
lines.append(f"- Annualised Sharpe (rf = 0): **{sharpe:.2f}**")
lines.append(f"- Max drawdown: **{max_dd:.2%}**")
lines.append(f"- Total costs paid: {total_cost:.2%} of capital")
lines.append(f"- Gross return BEFORE costs: {gross_total:+.2%} (Sharpe {gross_sharpe:.2f}) — "
             f"so costs, not the raw signal alone, do most of the damage: the "
             f"technical-only signal flips holdings almost every week and the "
             f"200 bps switch charge compounds against it")
lines.append(f"- Weekly rebalances in test period: {n_weeks}; weeks fully in cash: {cash_weeks} "
             f"({cash_weeks / weeks_total:.0%} of weeks)")
lines.append(f"- Classifier direction accuracy (weekly, 5-day outcome): {accuracy:.1%} "
             f"({len(pred_df)} predictions)")
lines.append(f"- Training rows at first fold: " +
             ", ".join(f"{t}={v}" for t, v in
                       [(t, int(frames[t].iloc[:fold_starts[0] - 5 + 1].dropna(subset=FEATURES + ['label']).shape[0])) for t in TICKERS]) +
             " (thin — the 200-day MA needs ~9.5 months of history; recorded, not fixed)")
lines.append("")
lines.append("## Per-fold results\n")
lines.append("| Fold | Start | End | Days | Return | Sharpe | Max DD |")
lines.append("|---:|---|---|---:|---:|---:|---:|")
for f in fold_stats:
    tag = " ← HOLDOUT" if f["holdout"] else ""
    lines.append(f"| {f['fold']} | {f['start']} | {f['end']} | {f['days']} | "
                 f"{f['return']:+.2%} | {f['sharpe']:.2f} | {f['max_dd']:.2%} |{tag}")
lines.append("")
lines.append(f"- **Worst fold:** fold {worst['fold']} ({worst['start']} → {worst['end']}), "
             f"return {worst['return']:+.2%}, Sharpe {worst['sharpe']:.2f}, max DD {worst['max_dd']:.2%}")
lines.append("")
lines.append("## Holdout fold (final 3 months, opened once, reported separately)\n")
lines.append(f"- Fold {holdout['fold']}: {holdout['start']} → {holdout['end']}, "
             f"return {holdout['return']:+.2%}, Sharpe {holdout['sharpe']:.2f}, "
             f"max DD {holdout['max_dd']:.2%}")
lines.append("")
lines.append("## Monte Carlo (1,000 block-bootstraps, 5-day blocks, seed 42)\n")
lines.append(f"- Observed Sharpe: {sharpe:.2f}")
lines.append(f"- 95th percentile of reshuffled Sharpes: {p95:.2f}")
lines.append(f"- **p = {p_value:.3f}** (share of reshuffled Sharpes ≥ observed)")
lines.append("")
lines.append("## Gates\n")
lines.append("| Gate | Value | Threshold | Result |")
lines.append("|---|---|---|---|")
for name, val, thr, ok in gates:
    v = f"{val:.2%}" if "drawdown" in name.lower() or "return" in name.lower() else f"{val:.2f}" if "Sharpe" in name else f"{val:.3f}"
    lines.append(f"| {name} | {v} | {thr} | {'**PASS**' if ok else '**FAIL**'} |")
lines.append("")
lines.append("## Benchmark (context only, NOT a gate)\n")
lines.append(f"- Equal-weight buy-and-hold, same period, same 200 bps cost rule: "
             f"total return {b_total:+.2%}, Sharpe {b_sharpe:.2f}, max drawdown {b_dd:.2%}")
lines.append("")
lines.append("## What this means\n")
if passed:
    lines.append("The rule survived a multi-year walk-forward at double costs with "
                 "a Monte Carlo reshuffle. It is a backtest, not a promise of future "
                 "returns, and not financial advice. Deployment is in your OWN "
                 "account only — prop firms ban third-party trading.")
else:
    lines.append("The rule did NOT survive the pre-registered gates. The most likely "
                 "reading: the paper's 5.65 Sharpe was a single 14-month window "
                 "(Mar 2025–Apr 2026) — exactly the limitation its author flagged. "
                 "On a longer multi-year sample at double costs, the edge does not "
                 "clear the bar. The sketch stays a hypothesis; it is frozen, not "
                 "re-tuned. The next candidates in the queue are Sketch 6 "
                 "(news-echo ORB) and Sketch 8 (BOCPD regime gate).")
lines.append("")
lines.append("## Reproducibility\n")
lines.append("- Data: `verified/data/{BUG,CLOU,BLOK,BOTZ}.csv` — Yahoo Finance chart "
             "API, adjusted closes, 1 Nov 2019 → 9 Nov 2026 (1,764 rows each).")
lines.append("- Script: `verified/run_sketch10_test.py` (BaggingClassifier, "
             "DecisionTreeClassifier base, 100 estimators, random_state=42; "
             "Monte Carlo seed 42).")
lines.append("- Costs: 200 bps per one-way switch, charged on the rebalance date; "
             "cash earns 0%; no leverage, no shorting.")
lines.append("- Implementation note: the first run charged a phantom full-entry "
             "cost at every fold boundary (position reset to cash per fold). "
             "That was a bug, not a gate change — the pre-registered rule "
             "charges costs on ACTUAL switches, so the position now carries "
             "across folds and this report is from the corrected run.")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print("\nWROTE:", OUT)
