"""Comprehensive strategy validation suite.

Tests:
  1. Intraday 5min RSI strategy backtest
  2. Rolling Walk-Forward for swing strategies (3yr train + 1yr test x N windows)
  3. Monte Carlo shuffle test (randomize trade order, check if edge survives)
  4. Parameter sensitivity analysis (neighbor stability check)

Usage:
    python run_validation.py
    python run_validation.py --test intraday
    python run_validation.py --test walkforward
    python run_validation.py --test montecarlo
    python run_validation.py --test sensitivity
"""

import sys
import io
import argparse
import itertools
import math
import time
from datetime import datetime, timedelta

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from data.indicators import TechnicalIndicators
from utils.helpers import load_yaml, get_project_root


# ═══════════════════════════════════════════════════════════════
#  Shared utilities
# ═══════════════════════════════════════════════════════════════

def fetch_daily_data(symbols: list[str], start: str = "2015-01-01") -> dict[str, pd.DataFrame]:
    from data.history import HistoryManager
    root = get_project_root()
    settings = load_yaml(str(root / "config" / "settings.yaml"))
    hm = HistoryManager()
    result = {}
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        from futu import OpenQuoteContext, RET_OK, KLType
        ctx = OpenQuoteContext(host=settings["futu"]["host"], port=settings["futu"]["port"])
    except Exception as e:
        print(f"[WARN] FutuOpenD unavailable ({e}), using cache only")
        ctx = None

    for sym in symbols:
        cached = hm.load_from_cache(sym, "K_DAY")
        if cached is not None and len(cached) >= 2000:
            result[sym] = cached
            print(f"  {sym}: {len(cached)} daily bars (cached)")
            continue
        if ctx is None:
            continue
        time.sleep(0.3)
        all_pages = []
        page_key = None
        while True:
            kwargs = dict(code=sym, start=start, end=end_date, ktype=KLType.K_DAY, max_count=1000)
            if page_key is not None:
                kwargs["page_req_key"] = page_key
            ret, data, page_key = ctx.request_history_kline(**kwargs)
            if ret == RET_OK and data is not None and len(data) > 0:
                all_pages.append(data)
            else:
                break
            if page_key is None:
                break
            time.sleep(0.3)
        if all_pages:
            df = pd.concat(all_pages, ignore_index=True).drop_duplicates(
                subset=["time_key"], keep="last"
            ).sort_values("time_key").reset_index(drop=True)
            hm.save_to_cache(sym, "K_DAY", df)
            result[sym] = df
            print(f"  {sym}: {len(df)} daily bars (API)")

    if ctx is not None:
        ctx.close()
    return result


def fetch_5min_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch as much 5-min historical data as possible (Futu limits ~1000 bars ≈ 2 weeks)."""
    root = get_project_root()
    settings = load_yaml(str(root / "config" / "settings.yaml"))
    result = {}

    try:
        from futu import OpenQuoteContext, RET_OK, KLType
        ctx = OpenQuoteContext(host=settings["futu"]["host"], port=settings["futu"]["port"])
    except Exception as e:
        print(f"[WARN] FutuOpenD unavailable ({e})")
        return result

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    for sym in symbols:
        time.sleep(0.3)
        all_pages = []
        page_key = None
        while True:
            kwargs = dict(code=sym, start=start_date, end=end_date, ktype=KLType.K_5M, max_count=1000)
            if page_key is not None:
                kwargs["page_req_key"] = page_key
            ret, data, page_key = ctx.request_history_kline(**kwargs)
            if ret == RET_OK and data is not None and len(data) > 0:
                all_pages.append(data)
            else:
                break
            if page_key is None:
                break
            time.sleep(0.3)
        if all_pages:
            df = pd.concat(all_pages, ignore_index=True).drop_duplicates(
                subset=["time_key"], keep="last"
            ).sort_values("time_key").reset_index(drop=True)
            result[sym] = df
            print(f"  {sym}: {len(df)} 5min bars")
        else:
            print(f"  {sym}: no 5min data")

    ctx.close()
    return result


def precompute_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for p in [5, 8, 10, 14, 15, 20]:
        col = f"ma_{p}"
        if col not in out.columns:
            out[col] = out["close"].rolling(p).mean()
    for p in [5, 8, 15, 20]:
        col = f"ema_{p}"
        if col not in out.columns:
            out[col] = out["close"].ewm(span=p, adjust=False).mean()
    for p in [5, 7, 10, 14]:
        col = f"rsi_{p}"
        if col not in out.columns:
            delta = out["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_g = gain.rolling(p).mean()
            avg_l = loss.rolling(p).mean()
            rs = avg_g / avg_l.replace(0, np.inf)
            out[col] = 100 - (100 / (1 + rs))
    for bp in [15, 20]:
        for bs in [1.5, 2.0]:
            tag = f"bb_{bp}_{bs}"
            if f"{tag}_upper" not in out.columns:
                sma = out["close"].rolling(bp).mean()
                std = out["close"].rolling(bp).std()
                out[f"{tag}_upper"] = sma + bs * std
                out[f"{tag}_middle"] = sma
                out[f"{tag}_lower"] = sma - bs * std
    for p in [14]:
        col = f"atr_{p}"
        if col not in out.columns:
            out = TechnicalIndicators.add_atr(out, p)
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def fast_backtest(signals, strengths, closes, initial_capital=3000.0,
                  commission_pct=0.001, slippage_pct=0.0005):
    capital = initial_capital
    position = 0
    avg_entry = 0.0
    trades = []
    equity = np.empty(len(closes))

    for i in range(len(closes)):
        price = closes[i]
        sig = signals[i]
        if sig == 1 and position == 0:
            buy_p = price * (1 + slippage_pct)
            qty = int(capital * 0.95 / buy_p)
            if qty > 0:
                comm = buy_p * qty * commission_pct
                cost = buy_p * qty + comm
                if cost <= capital:
                    position = qty
                    avg_entry = buy_p
                    capital -= cost
                    trades.append({"type": "BUY", "price": buy_p, "qty": qty, "bar": i})
        elif sig == -1 and position > 0:
            sell_p = price * (1 - slippage_pct)
            comm = sell_p * position * commission_pct
            revenue = sell_p * position - comm
            pnl = (sell_p - avg_entry) * position - comm
            capital += revenue
            trades.append({"type": "SELL", "price": sell_p, "qty": position, "pnl": pnl, "bar": i})
            position = 0
            avg_entry = 0.0
        equity[i] = capital + position * price

    final = capital + position * closes[-1]
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    return {"initial": initial_capital, "final": final, "trades": trades,
            "equity": equity, "sell_trades": sell_trades}


def calc_sharpe(equity):
    if len(equity) < 2:
        return 0.0
    rets = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    rf = (1.05 ** (1 / 252) - 1)
    excess = rets - rf
    std = float(np.std(rets, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(252))


def calc_max_dd(equity):
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    return float(np.max(dd)) * 100


# Re-wrap stdout after importing run_param_scan (it also wraps stdout at module level)
_orig_stdout = sys.stdout
from run_param_scan import (
    gen_momentum_signals, gen_mean_reversion_signals,
    gen_breakout_signals, gen_rsi_reversal_signals,
)
if hasattr(sys.stdout, 'closed') and sys.stdout.closed:
    sys.stdout = io.TextIOWrapper(sys.__stdout__.buffer, encoding="utf-8", errors="replace")

SIGNAL_GENERATORS = {
    "momentum": gen_momentum_signals,
    "mean_reversion": gen_mean_reversion_signals,
    "breakout": gen_breakout_signals,
    "rsi_reversal": gen_rsi_reversal_signals,
}

BEST_PARAMS = {
    "momentum": {"fast_ma": 8, "slow_ma": 15, "rsi_period": 10, "vol_threshold": 1.0},
    "mean_reversion": {"bb_period": 15, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75},
    "breakout": {"lookback": 10, "vol_threshold": 1.2, "atr_mult": 1.5},
    "rsi_reversal": {"rsi_period": 5, "rsi_buy": 25, "rsi_sell": 75},
}


# ═══════════════════════════════════════════════════════════════
#  TEST 1: Intraday 5min RSI strategy backtest
# ═══════════════════════════════════════════════════════════════

def gen_intraday_signals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the live intraday entry/exit logic on 5min bars."""
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)

    rsi5 = df["rsi_5"].values if "rsi_5" in df.columns else np.full(n, np.nan)
    ema8 = df["ema_8"].values if "ema_8" in df.columns else np.full(n, np.nan)
    ema20 = df["ema_20"].values if "ema_20" in df.columns else np.full(n, np.nan)
    bb_lower = df["bb_lower"].values if "bb_lower" in df.columns else np.full(n, np.nan)
    close = df["close"].values

    in_position = False
    entry_price = 0.0

    for i in range(20, n):
        if np.isnan(rsi5[i]) or np.isnan(rsi5[i - 1]):
            continue

        if not in_position:
            # Entry 1: RSI oversold bounce
            if 5 < rsi5[i - 1] < 25 <= rsi5[i] < 50:
                strength = min(60 + (25 - rsi5[i - 1]) * 1.5, 90)
                if not np.isnan(ema8[i]) and not np.isnan(ema20[i]) and ema8[i] > ema20[i]:
                    strength = min(strength + 10, 95)
                signals[i] = 1
                strengths[i] = strength
                in_position = True
                entry_price = close[i]
                continue

            # Entry 2: BB lower + RSI < 30
            if not np.isnan(bb_lower[i]) and close[i] <= bb_lower[i] and rsi5[i] < 30:
                strength = min(55 + (30 - rsi5[i]) * 1.2, 85)
                signals[i] = 1
                strengths[i] = strength
                in_position = True
                entry_price = close[i]
                continue
        else:
            pnl_pct = (close[i] / entry_price - 1) * 100 if entry_price > 0 else 0

            # Exit: RSI overbought
            if rsi5[i] > 70:
                signals[i] = -1
                strengths[i] = 70
                in_position = False
                continue

            # Exit: target profit
            if pnl_pct >= 2.0:
                signals[i] = -1
                strengths[i] = 65
                in_position = False
                continue

            # Exit: stop loss
            if pnl_pct <= -1.5:
                signals[i] = -1
                strengths[i] = 60
                in_position = False
                continue

    return signals, strengths


def test_intraday():
    print("\n" + "=" * 70)
    print("TEST 1: Intraday 5min RSI Strategy Backtest")
    print("=" * 70)

    symbols = ["US.TQQQ", "US.SOXL", "US.TNA"]
    print("\nFetching 5min data...")
    data_5m = fetch_5min_data(symbols)

    if not data_5m:
        print("ERROR: No 5min data available. Skipping intraday test.")
        return {}

    results = {}
    for sym, df in data_5m.items():
        df = TechnicalIndicators.add_rsi(df.copy(), 5)
        df = TechnicalIndicators.add_ema(df, 8)
        df = TechnicalIndicators.add_ema(df, 20)
        df = TechnicalIndicators.add_bollinger(df, 20, 2.0)

        sigs, strs = gen_intraday_signals(df)
        closes = df["close"].values
        bt = fast_backtest(sigs, strs, closes)

        n_trades = len(bt["sell_trades"])
        ret_pct = (bt["final"] / bt["initial"] - 1) * 100
        sharpe = calc_sharpe(bt["equity"])
        max_dd = calc_max_dd(bt["equity"])
        wins = [t["pnl"] for t in bt["sell_trades"] if t.get("pnl", 0) > 0]
        wr = len(wins) / n_trades * 100 if n_trades > 0 else 0

        days = len(df) / (78)  # ~78 bars per day on 5min
        results[sym] = {
            "bars": len(df), "days": f"{days:.0f}", "trades": n_trades,
            "return_pct": ret_pct, "sharpe": sharpe, "max_dd": max_dd, "win_rate": wr,
        }

        print(f"\n  {sym}: {len(df)} bars (~{days:.0f} trading days)")
        print(f"    Trades: {n_trades}")
        print(f"    Return: {ret_pct:+.2f}%")
        print(f"    Sharpe: {sharpe:.4f}")
        print(f"    Max DD: {max_dd:.2f}%")
        print(f"    Win Rate: {wr:.1f}%")

        if n_trades == 0:
            print(f"    WARNING: No trades generated! Strategy may not trigger on this data range.")

    return results


# ═══════════════════════════════════════════════════════════════
#  TEST 2: Rolling Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════

def test_rolling_walkforward():
    print("\n" + "=" * 70)
    print("TEST 2: Rolling Walk-Forward Validation (3yr train + 1yr test)")
    print("=" * 70)

    symbols = ["US.TQQQ", "US.SOXL", "US.TNA"]
    print("\nFetching daily data...")
    all_data = fetch_daily_data(symbols)

    if not all_data:
        print("ERROR: No daily data. Skipping.")
        return {}

    train_bars = 252 * 3  # 3 years
    test_bars = 252       # 1 year
    step_bars = 252       # slide 1 year at a time

    results = {}

    for strat_name in ["momentum", "mean_reversion", "rsi_reversal"]:
        gen_fn = SIGNAL_GENERATORS[strat_name]
        params = BEST_PARAMS[strat_name]

        print(f"\n  Strategy: {strat_name} | params={params}")

        for sym, raw_df in all_data.items():
            df = precompute_daily(raw_df)
            n = len(df)
            closes = df["close"].values

            windows = []
            start_idx = 0
            while start_idx + train_bars + test_bars <= n:
                train_end = start_idx + train_bars
                test_end = train_end + test_bars

                train_df = df.iloc[start_idx:train_end].reset_index(drop=True)
                test_df = df.iloc[train_end:test_end].reset_index(drop=True)

                # Train
                train_sigs, train_strs = gen_fn(train_df, params)
                train_bt = fast_backtest(train_sigs, train_strs, train_df["close"].values)
                train_sharpe = calc_sharpe(train_bt["equity"])
                train_ret = (train_bt["final"] / train_bt["initial"] - 1) * 100

                # Test (out of sample)
                test_sigs, test_strs = gen_fn(test_df, params)
                test_bt = fast_backtest(test_sigs, test_strs, test_df["close"].values)
                test_sharpe = calc_sharpe(test_bt["equity"])
                test_ret = (test_bt["final"] / test_bt["initial"] - 1) * 100
                test_trades = len(test_bt["sell_trades"])

                overfit = (1 - test_sharpe / train_sharpe) * 100 if train_sharpe != 0 else None

                train_start_date = train_df["time_key"].iloc[0][:10] if "time_key" in train_df.columns else f"bar_{start_idx}"
                test_end_date = test_df["time_key"].iloc[-1][:10] if "time_key" in test_df.columns else f"bar_{test_end}"

                windows.append({
                    "period": f"{train_start_date} -> {test_end_date}",
                    "train_sharpe": train_sharpe,
                    "train_ret": train_ret,
                    "test_sharpe": test_sharpe,
                    "test_ret": test_ret,
                    "test_trades": test_trades,
                    "overfit_pct": overfit,
                })

                start_idx += step_bars

            if windows:
                key = f"{strat_name}@{sym}"
                results[key] = windows

                avg_train_sharpe = np.mean([w["train_sharpe"] for w in windows])
                avg_test_sharpe = np.mean([w["test_sharpe"] for w in windows])
                avg_overfit = np.mean([w["overfit_pct"] for w in windows if w["overfit_pct"] is not None])
                consistent = sum(1 for w in windows if w["test_sharpe"] > 0) / len(windows) * 100

                print(f"    {sym}: {len(windows)} windows")
                print(f"      Avg train Sharpe: {avg_train_sharpe:.4f}")
                print(f"      Avg test  Sharpe: {avg_test_sharpe:.4f}")
                print(f"      Avg overfit:      {avg_overfit:.1f}%")
                print(f"      Consistency:      {consistent:.0f}% windows profitable")
                for w in windows:
                    of = f"{w['overfit_pct']:+.0f}%" if w["overfit_pct"] is not None else "N/A"
                    print(f"        {w['period']}  train={w['train_sharpe']:.3f} test={w['test_sharpe']:.3f} overfit={of} trades={w['test_trades']}")

    return results


# ═══════════════════════════════════════════════════════════════
#  TEST 3: Monte Carlo Shuffle Test
# ═══════════════════════════════════════════════════════════════

def test_montecarlo():
    print("\n" + "=" * 70)
    print("TEST 3: Monte Carlo Trade Shuffle (1000 permutations)")
    print("=" * 70)

    symbols = ["US.TQQQ", "US.SOXL", "US.TNA"]
    print("\nFetching daily data...")
    all_data = fetch_daily_data(symbols)

    if not all_data:
        print("ERROR: No data. Skipping.")
        return {}

    n_sims = 1000
    results = {}

    for strat_name in ["momentum", "rsi_reversal"]:
        gen_fn = SIGNAL_GENERATORS[strat_name]
        params = BEST_PARAMS[strat_name]

        for sym, raw_df in all_data.items():
            df = precompute_daily(raw_df)
            sigs, strs = gen_fn(df, params)
            closes = df["close"].values
            bt = fast_backtest(sigs, strs, closes)

            actual_pnls = [t["pnl"] for t in bt["sell_trades"] if "pnl" in t]
            if len(actual_pnls) < 3:
                continue

            actual_total = sum(actual_pnls)
            actual_sharpe = calc_sharpe(bt["equity"])

            # Shuffle PnLs and rebuild equity
            rng = np.random.default_rng(42)
            sim_sharpes = []
            sim_totals = []
            for _ in range(n_sims):
                shuffled = rng.permutation(actual_pnls)
                equity = np.zeros(len(shuffled) + 1)
                equity[0] = 3000.0
                for j, pnl in enumerate(shuffled):
                    equity[j + 1] = equity[j] + pnl
                sim_sharpes.append(calc_sharpe(equity))
                sim_totals.append(sum(shuffled))

            # What percentile is our actual Sharpe among random orderings?
            pctile = np.mean([1 for s in sim_sharpes if actual_sharpe > s]) * 100

            key = f"{strat_name}@{sym}"
            results[key] = {
                "actual_sharpe": actual_sharpe,
                "actual_total_pnl": actual_total,
                "mc_mean_sharpe": np.mean(sim_sharpes),
                "mc_p5_sharpe": np.percentile(sim_sharpes, 5),
                "mc_p95_sharpe": np.percentile(sim_sharpes, 95),
                "percentile": pctile,
                "n_trades": len(actual_pnls),
            }

            print(f"\n  {strat_name}@{sym}: {len(actual_pnls)} trades")
            print(f"    Actual Sharpe:  {actual_sharpe:.4f}")
            print(f"    MC mean Sharpe: {np.mean(sim_sharpes):.4f}")
            print(f"    MC 5th-95th:    [{np.percentile(sim_sharpes, 5):.4f}, {np.percentile(sim_sharpes, 95):.4f}]")
            print(f"    Percentile:     {pctile:.0f}th (higher=better, >50=edge exists)")

    return results


# ═══════════════════════════════════════════════════════════════
#  TEST 4: Parameter Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════

def test_sensitivity():
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Sensitivity Analysis")
    print("=" * 70)

    symbols = ["US.TQQQ", "US.SOXL"]
    print("\nFetching daily data...")
    all_data = fetch_daily_data(symbols)

    if not all_data:
        print("ERROR: No data. Skipping.")
        return {}

    # For momentum: vary each param ±1 step from optimal
    param_neighbors = {
        "momentum": {
            "fast_ma": [6, 7, 8, 9, 10],
            "slow_ma": [12, 13, 15, 17, 20],
            "rsi_period": [8, 9, 10, 11, 12],
            "vol_threshold": [0.8, 0.9, 1.0, 1.1, 1.2],
        },
        "rsi_reversal": {
            "rsi_period": [4, 5, 6, 7],
            "rsi_buy": [20, 22, 25, 28, 30],
            "rsi_sell": [70, 72, 75, 78, 80],
        },
    }

    results = {}

    for strat_name, neighbor_grid in param_neighbors.items():
        gen_fn = SIGNAL_GENERATORS[strat_name]
        base_params = BEST_PARAMS[strat_name].copy()

        for sym, raw_df in all_data.items():
            df = precompute_daily(raw_df)
            closes = df["close"].values

            # Baseline
            base_sigs, base_strs = gen_fn(df, base_params)
            base_bt = fast_backtest(base_sigs, base_strs, closes)
            base_sharpe = calc_sharpe(base_bt["equity"])
            base_ret = (base_bt["final"] / base_bt["initial"] - 1) * 100

            print(f"\n  {strat_name}@{sym} baseline: Sharpe={base_sharpe:.4f} Ret={base_ret:+.1f}%")

            for param_name, values in neighbor_grid.items():
                sharpes = []
                for val in values:
                    test_params = base_params.copy()
                    test_params[param_name] = val
                    try:
                        sigs, strs = gen_fn(df, test_params)
                        bt = fast_backtest(sigs, strs, closes)
                        s = calc_sharpe(bt["equity"])
                    except Exception:
                        s = 0.0
                    sharpes.append(s)

                is_optimal = values[len(values) // 2] == base_params[param_name]
                std_dev = np.std(sharpes)
                mean_sharpe = np.mean(sharpes)
                stability = "STABLE" if std_dev < 0.3 else ("MODERATE" if std_dev < 0.6 else "UNSTABLE")

                key = f"{strat_name}@{sym}:{param_name}"
                results[key] = {
                    "values": values,
                    "sharpes": [round(s, 4) for s in sharpes],
                    "std": std_dev,
                    "stability": stability,
                }

                val_str = "  ".join(f"{v}={s:.3f}" for v, s in zip(values, sharpes))
                print(f"    {param_name}: [{val_str}]  std={std_dev:.4f} -> {stability}")

    return results


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def print_summary(intraday_res, wf_res, mc_res, sens_res):
    print("\n\n" + "=" * 70)
    print("VALIDATION SUMMARY REPORT")
    print("=" * 70)

    print("\n1. INTRADAY 5min RSI STRATEGY:")
    if intraday_res:
        for sym, r in intraday_res.items():
            verdict = "PASS" if r["sharpe"] > 0 and r["trades"] >= 3 else "FAIL/INSUFFICIENT"
            print(f"   {sym}: Sharpe={r['sharpe']:.4f} Trades={r['trades']} WR={r['win_rate']:.0f}% -> {verdict}")
    else:
        print("   NOT TESTED (no 5min data)")

    print("\n2. ROLLING WALK-FORWARD:")
    if wf_res:
        for key, windows in wf_res.items():
            avg_test = np.mean([w["test_sharpe"] for w in windows])
            consist = sum(1 for w in windows if w["test_sharpe"] > 0) / len(windows) * 100
            verdict = "PASS" if consist >= 60 and avg_test > 0 else "WEAK" if consist >= 40 else "FAIL"
            print(f"   {key}: Avg OOS Sharpe={avg_test:.4f} Consistency={consist:.0f}% -> {verdict}")
    else:
        print("   NOT TESTED")

    print("\n3. MONTE CARLO:")
    if mc_res:
        for key, r in mc_res.items():
            verdict = "PASS" if r["percentile"] >= 50 else "WEAK" if r["percentile"] >= 30 else "FAIL"
            print(f"   {key}: Pctile={r['percentile']:.0f}th Actual={r['actual_sharpe']:.4f} -> {verdict}")
    else:
        print("   NOT TESTED")

    print("\n4. PARAMETER SENSITIVITY:")
    if sens_res:
        unstable_count = sum(1 for r in sens_res.values() if r["stability"] == "UNSTABLE")
        total = len(sens_res)
        print(f"   {total - unstable_count}/{total} parameters are STABLE or MODERATE")
        for key, r in sens_res.items():
            if r["stability"] == "UNSTABLE":
                print(f"   WARNING: {key} is UNSTABLE (std={r['std']:.4f})")
    else:
        print("   NOT TESTED")


def main():
    parser = argparse.ArgumentParser(description="FUTU-QUANT Strategy Validation Suite")
    parser.add_argument("--test", choices=["intraday", "walkforward", "montecarlo", "sensitivity", "all"],
                        default="all", help="Which test to run")
    args = parser.parse_args()

    intraday_res = {}
    wf_res = {}
    mc_res = {}
    sens_res = {}

    if args.test in ("intraday", "all"):
        intraday_res = test_intraday()

    if args.test in ("walkforward", "all"):
        wf_res = test_rolling_walkforward()

    if args.test in ("montecarlo", "all"):
        mc_res = test_montecarlo()

    if args.test in ("sensitivity", "all"):
        sens_res = test_sensitivity()

    print_summary(intraday_res, wf_res, mc_res, sens_res)

    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    with open("results/validation_report.txt", "w", encoding="utf-8") as f:
        import contextlib
        with contextlib.redirect_stdout(f):
            print_summary(intraday_res, wf_res, mc_res, sens_res)
    print(f"\nReport saved to results/validation_report.txt")


if __name__ == "__main__":
    main()
