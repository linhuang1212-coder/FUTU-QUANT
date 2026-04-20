"""Validate intraday strategies using REAL 5min data (575 days from Futu API).

This is the critical cross-validation: strategies were developed on synthetic
data, now we test on real market microstructure.
"""
import sys, io, math
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.downloader import load_5min
from data.indicators import TechnicalIndicators


def precompute(df):
    out = df.copy()
    out = TechnicalIndicators.add_rsi(out, 14)
    out = TechnicalIndicators.add_ema(out, 8)
    out = TechnicalIndicators.add_ema(out, 20)
    out = TechnicalIndicators.add_atr(out, 14)
    tp = (out["high"] + out["low"] + out["close"]) / 3
    cum_tp_vol = (tp * out["volume"]).cumsum()
    cum_vol = out["volume"].cumsum().replace(0, np.nan)
    out["vwap"] = cum_tp_vol / cum_vol
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def split_days(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["time_key"]).dt.date
    days = []
    for _, g in df.groupby("date"):
        d = g.reset_index(drop=True)
        if len(d) >= 20:
            days.append(d)
    return days


# ── Strategies ──

def sim_rebalance_2pm(day_df, params):
    move_t = params.get("move_threshold_pct", 1.5)
    entry_idx = params.get("entry_bar_idx", 48)
    stop_pct = params.get("stop_pct", 1.0)
    if len(day_df) <= entry_idx + 3:
        return None
    op = day_df["open"].iloc[0]
    p = day_df["close"].iloc[entry_idx]
    move = (p - op) / op * 100
    if move < move_t:
        return None
    ep = p * 1.0005
    sp = ep * (1 - stop_pct / 100)
    for i in range(entry_idx + 1, len(day_df)):
        if day_df["low"].iloc[i] <= sp:
            return (sp * 0.9995 - ep) / ep * 100
    return (day_df["close"].iloc[-1] * 0.9995 - ep) / ep * 100


def sim_afternoon_ext(day_df, params):
    return sim_rebalance_2pm(day_df, params)


def sim_vol_squeeze(day_df, params):
    squeeze_ratio = params.get("squeeze_ratio", 0.5)
    entry_after = params.get("entry_after_bar", 24)
    stop_pct = params.get("stop_pct", 1.5)
    if len(day_df) < entry_after + 10:
        return None
    f2h_high = day_df["high"].iloc[:entry_after].max()
    f2h_low = day_df["low"].iloc[:entry_after].min()
    f2h_range = f2h_high - f2h_low
    if f2h_range <= 0:
        return None
    avg_range = day_df["high"].mean() - day_df["low"].mean()
    if avg_range <= 0:
        return None
    if f2h_range / avg_range > squeeze_ratio * entry_after:
        return None
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    for i in range(entry_after, len(day_df)):
        c = day_df["close"].iloc[i]
        if position == 0:
            if c > f2h_high:
                entry_price = c * 1.0005
                stop_price = entry_price * (1 - stop_pct / 100)
                position = 1
        else:
            if day_df["low"].iloc[i] <= stop_price:
                return (stop_price * 0.9995 - entry_price) / entry_price * 100
    if position > 0:
        return (day_df["close"].iloc[-1] * 0.9995 - entry_price) / entry_price * 100
    return None


def sim_vwap_trend(day_df, params):
    confirm = params.get("confirm_bars", 3)
    rsi_floor = params.get("rsi_floor", 50)
    min_bars = params.get("min_bars_before_entry", 6)
    if len(day_df) < min_bars + confirm + 5 or "vwap" not in day_df.columns:
        return None
    close = day_df["close"].values
    vwap = day_df["vwap"].values
    rsi = day_df["rsi_14"].values if "rsi_14" in day_df.columns else np.full(len(day_df), 50)
    pos = 0
    ep = 0.0
    for i in range(min_bars, len(day_df)):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]):
            continue
        if pos == 0:
            above = sum(1 for k in range(confirm) if i - k >= 0 and close[i - k] > vwap[i - k])
            if above >= confirm and rsi[i] > rsi_floor:
                ep = close[i] * 1.0005
                pos = 1
        else:
            if close[i] < vwap[i]:
                return (close[i] * 0.9995 - ep) / ep * 100
    if pos > 0:
        return (day_df["close"].iloc[-1] * 0.9995 - ep) / ep * 100
    return None


# ── Backtest ──

def backtest(days, sim_fn, params, capital=3000.0):
    eq = capital
    trades = []
    equities = [capital]
    for d in days:
        pnl = sim_fn(d, params)
        if pnl is not None:
            eq += eq * 0.95 * pnl / 100
            trades.append(pnl)
        equities.append(eq)
    equities = np.array(equities)
    n = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = len(wins) / n * 100 if n > 0 else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0)
    ret = (eq / capital - 1) * 100
    if len(equities) > 1:
        rets = np.diff(equities) / np.where(equities[:-1] > 0, equities[:-1], 1)
        rf = 1.05 ** (1 / 252) - 1
        std = float(np.std(rets, ddof=1))
        sharpe = float(np.mean(rets - rf) / std * math.sqrt(252)) if std > 1e-12 else 0
        peak = np.maximum.accumulate(equities)
        dd = (peak - equities) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd)) * 100
    else:
        sharpe = max_dd = 0
    return {
        "sharpe": round(sharpe, 4), "return_pct": round(ret, 2),
        "max_dd": round(max_dd, 2), "trades": n,
        "win_rate": round(wr, 1),
        "pf": round(pf, 3) if math.isfinite(pf) else pf,
    }


# ── Main ──

STRATEGIES = [
    ("Rebalance_2pm (1.5%/48bar/1%)", sim_rebalance_2pm,
     {"move_threshold_pct": 1.5, "entry_bar_idx": 48, "stop_pct": 1.0}),
    ("Rebalance_2pm (1.0%/48bar/1%)", sim_rebalance_2pm,
     {"move_threshold_pct": 1.0, "entry_bar_idx": 48, "stop_pct": 1.0}),
    ("Afternoon_Ext (1.5%/36bar/1%)", sim_afternoon_ext,
     {"move_threshold_pct": 1.5, "entry_bar_idx": 36, "stop_pct": 1.0}),
    ("Afternoon_Ext (1.5%/42bar/1%)", sim_afternoon_ext,
     {"move_threshold_pct": 1.5, "entry_bar_idx": 42, "stop_pct": 1.0}),
    ("Vol_Squeeze (0.3/24bar/1.5%)", sim_vol_squeeze,
     {"squeeze_ratio": 0.3, "entry_after_bar": 24, "stop_pct": 1.5}),
    ("Vol_Squeeze (0.5/24bar/1.0%)", sim_vol_squeeze,
     {"squeeze_ratio": 0.5, "entry_after_bar": 24, "stop_pct": 1.0}),
    ("Vol_Squeeze (0.5/18bar/1.5%)", sim_vol_squeeze,
     {"squeeze_ratio": 0.5, "entry_after_bar": 18, "stop_pct": 1.5}),
    ("Vol_Squeeze (0.4/24bar/2.0%)", sim_vol_squeeze,
     {"squeeze_ratio": 0.4, "entry_after_bar": 24, "stop_pct": 2.0}),
    ("VWAP_Trend (3bar/RSI50)", sim_vwap_trend,
     {"confirm_bars": 3, "rsi_floor": 50, "min_bars_before_entry": 6}),
]


def main():
    print("=" * 80)
    print("REAL 5min DATA VALIDATION (575 trading days, 2024-01 ~ 2026-04)")
    print("=" * 80)

    for sym in ["TQQQ", "SOXL"]:
        df = load_5min(sym)
        if df is None:
            print(f"\n{sym}: no 5min data")
            continue

        n_days = len(set(pd.to_datetime(df["time_key"]).dt.date))
        print(f"\n{'='*60}")
        print(f"{sym}: {len(df)} bars, {n_days} trading days")
        print(f"{'='*60}")

        df = precompute(df)
        all_days = split_days(df)
        print(f"  Split into {len(all_days)} complete days")

        # Full period
        print(f"\n  {'Strategy':<35} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'Trades':>7} {'WR':>6} {'PF':>6}")
        print(f"  {'-'*77}")

        for name, fn, params in STRATEGIES:
            m = backtest(all_days, fn, params)
            print(f"  {name:<35} {m['sharpe']:>7.2f} {m['return_pct']:>7.1f}% {m['max_dd']:>6.1f}% {m['trades']:>7} {m['win_rate']:>5.1f}% {m['pf']:>6.2f}")

        # Walk-forward: first 60% train, last 40% test
        split = int(len(all_days) * 0.6)
        train_days = all_days[:split]
        test_days = all_days[split:]
        print(f"\n  Walk-Forward: train {len(train_days)} days, test {len(test_days)} days")
        print(f"  {'Strategy':<35} {'Train_Sh':>9} {'Test_Sh':>8} {'Test_Ret':>9} {'Test_Tr':>8} {'Verdict':>8}")
        print(f"  {'-'*80}")

        for name, fn, params in STRATEGIES:
            tr = backtest(train_days, fn, params)
            te = backtest(test_days, fn, params)
            verdict = "PASS" if te["sharpe"] > 0.5 and te["trades"] >= 3 else "FAIL"
            print(f"  {name:<35} {tr['sharpe']:>9.2f} {te['sharpe']:>8.2f} {te['return_pct']:>8.1f}% {te['trades']:>8} {verdict:>8}")


if __name__ == "__main__":
    main()
