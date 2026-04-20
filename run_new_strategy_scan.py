"""Fast vectorized scan + rolling WF validation for 4 new strategies.

Reuses fetch / precompute / fast_backtest from run_param_scan,
adds NumPy signal generators for:
  - vwap_reversion
  - multi_factor
  - volatility_breakout
  - fast_ema_cross

Usage:  python run_new_strategy_scan.py
"""

import sys, io, os, itertools, math
from pathlib import Path

import numpy as np
import pandas as pd

_saved_fd = os.dup(1)

from run_param_scan import (
    fetch_all_data,
    precompute_indicators,
    fast_backtest,
    compute_metrics,
)

sys.stdout = io.TextIOWrapper(os.fdopen(_saved_fd, "wb"), encoding="utf-8", errors="replace")

SYMBOLS = ["US.TQQQ", "US.SOXL"]

# ── extra pre-compute columns needed by new strategies ──────────────

def extend_precompute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for p in [3, 5, 7, 8, 10, 14, 15, 18, 20, 21]:
        col = f"ema_{p}"
        if col not in out.columns:
            out[col] = out["close"].ewm(span=p, adjust=False).mean()
        mcol = f"ma_{p}"
        if mcol not in out.columns:
            out[mcol] = out["close"].rolling(p).mean()

    if "macd_hist" not in out.columns:
        ema12 = out["close"].ewm(span=12, adjust=False).mean()
        ema26 = out["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = macd - signal

    tp = (out["high"] + out["low"] + out["close"]) / 3
    cum_tp_vol = (tp * out["volume"]).cumsum()
    cum_vol = out["volume"].cumsum()
    out["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    for rp in [5, 7, 10, 14]:
        col = f"rsi_{rp}"
        if col not in out.columns:
            delta = out["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_g = gain.rolling(rp).mean()
            avg_l = loss.rolling(rp).mean()
            rs = avg_g / avg_l.replace(0, np.inf)
            out[col] = 100 - (100 / (1 + rs))

    for bp in [15, 20]:
        for bs in [1.5, 2.0, 2.5]:
            tag = f"bb_{bp}_{bs}"
            if f"{tag}_upper" not in out.columns:
                sma = out["close"].rolling(bp).mean()
                std = out["close"].rolling(bp).std()
                out[f"{tag}_upper"] = sma + bs * std
                out[f"{tag}_middle"] = sma
                out[f"{tag}_lower"] = sma - bs * std

    if "vol_ma20" not in out.columns:
        out["vol_ma20"] = out["volume"].rolling(20).mean()

    return out


# ── Signal generators ───────────────────────────────────────────────

def gen_vwap_reversion_signals(df: pd.DataFrame, params: dict):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)

    dev = params["dev_threshold"]
    rsi_col = f"rsi_{params['rsi_period']}"
    rsi_os = params["rsi_oversold"]
    rsi_ob = params["rsi_overbought"]

    close = df["close"].values
    vwap = df["vwap"].values
    rsi = df[rsi_col].values
    vol = df["volume"].values
    vol_ma = df["vol_ma20"].values

    in_pos = False
    warmup = max(20, params["rsi_period"]) + 6

    for i in range(warmup, n):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]) or vwap[i] == 0 or vol_ma[i] <= 0:
            continue
        if vol[i] / vol_ma[i] < params.get("min_volume_ratio", 1.0):
            continue

        lower = vwap[i] * (1 - dev / 100)
        upper = vwap[i] * (1 + dev / 100)
        dist = (close[i] - vwap[i]) / vwap[i] * 100

        if not in_pos:
            if close[i] < lower and rsi[i] < rsi_os:
                excess = max(0, -dist - dev)
                s = min(95, 52 + excess * 3.5 + max(0, rsi_os - rsi[i]) * 0.35)
                signals[i] = 1
                strengths[i] = s
                in_pos = True
        else:
            if close[i] > upper and rsi[i] > rsi_ob:
                signals[i] = -1
                strengths[i] = 70
                in_pos = False
            elif i >= 2 and close[i-1] < vwap[i-1] and close[i] >= vwap[i]:
                was_os = False
                for k in range(max(warmup, i-5), i):
                    if close[k] < vwap[k] * (1 - dev / 100) and rsi[k] < rsi_os:
                        was_os = True
                        break
                if was_os:
                    signals[i] = -1
                    strengths[i] = 65
                    in_pos = False

    return signals, strengths


def gen_multi_factor_signals(df: pd.DataFrame, params: dict):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)

    fast_col = f"ma_{params['fast_ma_period']}"
    slow_col = f"ma_{params['slow_ma_period']}"
    rsi_col = f"rsi_{params['rsi_period']}"
    ema_col = f"ema_{params.get('ema_period', 20)}"
    buy_th = params["buy_threshold"]
    sell_th = params["sell_threshold"]

    fast_ma = df[fast_col].values
    slow_ma = df[slow_col].values
    rsi = df[rsi_col].values
    macd_h = df["macd_hist"].values
    close = df["close"].values
    ema = df[ema_col].values
    vol = df["volume"].values
    vol_ma = df["vol_ma20"].values

    warmup = max(params["slow_ma_period"], params["fast_ma_period"],
                 params["rsi_period"], 35, 20) + 5

    in_pos = False
    for i in range(warmup, n):
        if any(np.isnan(x) for x in (fast_ma[i], slow_ma[i], rsi[i], macd_h[i], ema[i])):
            continue
        vote = 0
        if fast_ma[i] > slow_ma[i]: vote += 1
        elif fast_ma[i] < slow_ma[i]: vote -= 1
        if rsi[i] > 50: vote += 1
        elif rsi[i] < 50: vote -= 1
        if macd_h[i] > 0: vote += 1
        elif macd_h[i] < 0: vote -= 1
        if close[i] > ema[i]: vote += 1
        elif close[i] < ema[i]: vote -= 1
        if vol_ma[i] > 0 and vol[i] > vol_ma[i]:
            vote += 1

        if not in_pos:
            if vote >= buy_th:
                signals[i] = 1
                strengths[i] = min(50 + abs(vote) * 10, 95)
                in_pos = True
        else:
            if vote <= -sell_th:
                signals[i] = -1
                strengths[i] = min(50 + abs(vote) * 10, 95)
                in_pos = False

    return signals, strengths


def gen_volatility_breakout_signals(df: pd.DataFrame, params: dict):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)

    bp = params["bb_period"]
    bs = params.get("bb_std", 2.0)
    sq_pct = params["squeeze_percentile"]
    sq_look = params["squeeze_lookback"]
    vt = params["vol_threshold"]
    rsi_p = params["rsi_period"]
    rsi_exit = params["rsi_exit"]
    width_lb = params["width_lookback"]

    tag = f"bb_{bp}_{bs}"
    upper = df[f"{tag}_upper"].values
    lower = df[f"{tag}_lower"].values
    middle = df[f"{tag}_middle"].values
    rsi_col = f"rsi_{rsi_p}"
    rsi = df[rsi_col].values
    close = df["close"].values
    vol = df["volume"].values
    vol_ma = df["vol_ma20"].values

    bb_width = np.where(middle > 0, (upper - lower) / middle, np.nan)
    in_squeeze = np.zeros(n, dtype=bool)
    for i in range(width_lb + bp, n):
        window = bb_width[i - width_lb:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= width_lb // 2:
            threshold = np.percentile(valid, sq_pct)
            if not np.isnan(bb_width[i]) and bb_width[i] <= threshold:
                in_squeeze[i] = True

    warmup = width_lb + bp + 5
    in_pos = False

    for i in range(warmup, n):
        if np.isnan(rsi[i]) or vol_ma[i] <= 0:
            continue

        had_squeeze = np.any(in_squeeze[max(warmup, i - sq_look):i])

        if not in_pos:
            cross_up = (close[i] > upper[i] and close[i-1] <= upper[i-1]
                        and not np.isnan(upper[i]) and not np.isnan(upper[i-1]))
            vol_ok = vol[i] > vt * vol_ma[i]
            if had_squeeze and cross_up and vol_ok and rsi[i] > 50:
                vol_ratio = vol[i] / vol_ma[i]
                s = min(55 + (vol_ratio - vt) * 15 + (rsi[i] - 50) * 0.4, 95)
                signals[i] = 1
                strengths[i] = s
                in_pos = True
        else:
            cross_down = (close[i] < lower[i] and close[i-1] >= lower[i-1]
                          and not np.isnan(lower[i]) and not np.isnan(lower[i-1])
                          and close[i-1] > middle[i-1])
            if cross_down or rsi[i] < rsi_exit:
                signals[i] = -1
                strengths[i] = 65
                in_pos = False

    return signals, strengths


def gen_fast_ema_cross_signals(df: pd.DataFrame, params: dict):
    n = len(df)
    signals = np.zeros(n, dtype=int)
    strengths = np.zeros(n, dtype=float)

    fp = params["fast_ema_period"]
    mp = params["medium_ema_period"]
    sp = params["slow_ema_period"]
    rsi_p = params["rsi_period"]

    fast = df[f"ema_{fp}"].values
    med = df[f"ema_{mp}"].values
    slow = df[f"ema_{sp}"].values
    rsi = df[f"rsi_{rsi_p}"].values
    vol = df["volume"].values
    vol_ma = df["vol_ma20"].values
    close = df["close"].values

    rsi_buy_f = params["rsi_buy_floor"]
    rsi_sell_c = params["rsi_sell_ceiling"]
    warmup = max(sp, rsi_p, 20) + 3

    in_pos = False
    for i in range(warmup, n):
        if any(np.isnan(x) for x in (fast[i], med[i], slow[i], rsi[i], fast[i-1], med[i-1])):
            continue
        if vol_ma[i] <= 0:
            continue

        cross_up = fast[i-1] <= med[i-1] and fast[i] > med[i]
        cross_down = fast[i-1] >= med[i-1] and fast[i] < med[i]
        vol_ratio = vol[i] / vol_ma[i]

        if not in_pos:
            if cross_up and med[i] > slow[i] and rsi[i] > rsi_buy_f:
                v_th = params.get("volume_ratio_threshold", 1.0)
                if v_th > 0 and vol_ratio < v_th:
                    continue
                s = min(95, 55 + min(20, max(0, rsi[i] - rsi_buy_f) * 0.8))
                if vol_ratio > 1.2:
                    s = min(100, s + min(18, (vol_ratio - 1.2) * 35))
                signals[i] = 1
                strengths[i] = s
                in_pos = True
        else:
            if cross_down and rsi[i] < rsi_sell_c:
                signals[i] = -1
                strengths[i] = min(92, 52 + max(0, rsi_sell_c - rsi[i]) * 1.2)
                in_pos = False

    return signals, strengths


SIGNAL_GENERATORS = {
    "vwap_reversion": gen_vwap_reversion_signals,
    "multi_factor": gen_multi_factor_signals,
    "volatility_breakout": gen_volatility_breakout_signals,
    "fast_ema_cross": gen_fast_ema_cross_signals,
}

STRATEGY_GRIDS = {
    "vwap_reversion": {
        "dev_threshold": [1.5, 2.0, 2.5, 3.0],
        "rsi_period": [7, 10, 14],
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
        "min_volume_ratio": [0.8, 1.0],
    },
    "multi_factor": {
        "fast_ma_period": [5, 8, 10],
        "slow_ma_period": [15, 20],
        "rsi_period": [10, 14],
        "ema_period": [15, 20],
        "buy_threshold": [3, 4],
        "sell_threshold": [3, 4],
    },
    "volatility_breakout": {
        "bb_period": [15, 20],
        "bb_std": [2.0],
        "squeeze_percentile": [15, 20, 25],
        "squeeze_lookback": [8, 12],
        "vol_threshold": [1.0, 1.2, 1.5],
        "rsi_period": [14],
        "rsi_exit": [35, 40],
        "width_lookback": [100],
    },
    "fast_ema_cross": {
        "fast_ema_period": [3, 5],
        "medium_ema_period": [8, 10, 13],
        "slow_ema_period": [18, 21, 26],
        "rsi_period": [10, 14],
        "rsi_buy_floor": [35, 40, 45],
        "rsi_sell_ceiling": [55, 60, 65],
        "volume_ratio_threshold": [0.8, 1.0],
    },
}


# ── Walk-forward scan ───────────────────────────────────────────────

def walk_forward_scan(strat_name, df, symbol, param_grid, train_pct=0.7):
    gen_fn = SIGNAL_GENERATORS[strat_name]
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    split = int(len(df) * train_pct)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:].reset_index(drop=True)
    train_c = train_df["close"].values
    val_c = val_df["close"].values

    train_results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            sigs, strs = gen_fn(train_df, params)
        except (KeyError, IndexError):
            continue
        result = fast_backtest(sigs, strs, train_c)
        metrics = compute_metrics(result, train_c)
        if metrics["total_trades"] >= 2:
            train_results.append({"params": params, "metrics": metrics})

    if not train_results:
        return []
    train_results.sort(key=lambda x: x["metrics"]["sharpe_ratio"], reverse=True)

    val_results = []
    for rank, entry in enumerate(train_results[:10], 1):
        params = entry["params"]
        sigs, strs = gen_fn(val_df, params)
        result = fast_backtest(sigs, strs, val_c)
        m = compute_metrics(result, val_c)
        ts = entry["metrics"]["sharpe_ratio"]
        vs = m["sharpe_ratio"]
        overfit = round((1 - vs / ts) * 100, 1) if ts and ts != 0 else None

        val_results.append({
            "strategy": strat_name, "symbol": symbol, "rank": rank,
            "train_sharpe": ts, "val_sharpe": vs,
            "val_return": m["total_return_pct"], "val_max_dd": m["max_drawdown_pct"],
            "val_trades": m["total_trades"], "val_wr": m["win_rate_pct"],
            "val_pf": m.get("profit_factor", 0),
            "overfit_pct": overfit,
            "params": params,
        })
    return val_results


# ── Rolling Walk-Forward ────────────────────────────────────────────

def rolling_walkforward(strat_name, params, df, symbol):
    gen_fn = SIGNAL_GENERATORS[strat_name]
    train_bars = 252 * 3
    test_bars = 252
    step = 252
    n = len(df)
    windows = []
    start = 0
    while start + train_bars + test_bars <= n:
        train_end = start + train_bars
        test_end = train_end + test_bars
        tr = df.iloc[start:train_end].reset_index(drop=True)
        te = df.iloc[train_end:test_end].reset_index(drop=True)
        try:
            s1, st1 = gen_fn(tr, params)
            r1 = fast_backtest(s1, st1, tr["close"].values)
            m1 = compute_metrics(r1, tr["close"].values)
            s2, st2 = gen_fn(te, params)
            r2 = fast_backtest(s2, st2, te["close"].values)
            m2 = compute_metrics(r2, te["close"].values)
        except Exception:
            start += step
            continue
        windows.append({
            "train_sharpe": m1["sharpe_ratio"],
            "test_sharpe": m2["sharpe_ratio"],
            "test_trades": m2["total_trades"],
            "test_return": m2["total_return_pct"],
        })
        start += step
    return windows


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NEW STRATEGY SCAN + VALIDATION (Vectorized)")
    print("=" * 70)

    print("\n[1/5] Fetching data...")
    all_data = fetch_all_data(SYMBOLS)
    if not all_data:
        print("ERROR: No data"); return

    print("\n[2/5] Precomputing indicators...")
    precomputed = {}
    for sym, df in all_data.items():
        base = precompute_indicators(df)
        precomputed[sym] = extend_precompute(base)
        print(f"  {sym}: {len(df)} bars, {len(precomputed[sym].columns)} cols")

    print("\n[3/5] Walk-Forward Parameter Scan...")
    all_results = []

    for strat_name, grid in STRATEGY_GRIDS.items():
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"\n  {strat_name} ({n_combos} combos)")

        for sym, df in precomputed.items():
            print(f"    {sym}...", end=" ", flush=True)
            try:
                results = walk_forward_scan(strat_name, df, sym, grid)
                all_results.extend(results)
                if results:
                    best = max(results, key=lambda r: r["val_sharpe"])
                    of = f"{best['overfit_pct']:+.0f}%" if best['overfit_pct'] is not None else "N/A"
                    print(f"Sharpe={best['val_sharpe']:.4f} Ret={best['val_return']:+.1f}% "
                          f"Tr={best['val_trades']} WR={best['val_wr']:.0f}% OF={of}")
                else:
                    print("no valid results")
            except Exception as e:
                print(f"error: {e}")

    good = [r for r in all_results if r["val_sharpe"] > 0.2 and r["val_trades"] >= 3]
    good.sort(key=lambda x: x["val_sharpe"], reverse=True)

    print(f"\n\n{'='*70}")
    print(f"[4/5] PROMISING (val_sharpe > 0.2, trades >= 3): {len(good)}")
    print("=" * 70)
    for r in good[:20]:
        of = f"{r['overfit_pct']:+.0f}%" if r['overfit_pct'] is not None else "N/A"
        print(f"  {r['strategy']:22} @ {r['symbol']:10} "
              f"Sh={r['val_sharpe']:.4f} Ret={r['val_return']:+.1f}% "
              f"DD={r['val_max_dd']:.1f}% Tr={r['val_trades']} WR={r['val_wr']:.0f}% "
              f"PF={r['val_pf']:.2f} OF={of}")
        print(f"    {r['params']}")

    print(f"\n\n{'='*70}")
    print("[5/5] Rolling Walk-Forward on Top Combos")
    print("=" * 70)

    validated = []
    for r in good[:15]:
        sym = r["symbol"]
        df = precomputed[sym]
        windows = rolling_walkforward(r["strategy"], r["params"], df, sym)
        if not windows:
            continue
        avg_test = np.mean([w["test_sharpe"] for w in windows])
        consistency = sum(1 for w in windows if w["test_sharpe"] > 0) / len(windows) * 100
        avg_trades = np.mean([w["test_trades"] for w in windows])

        verdict = "PASS" if consistency >= 60 and avg_test > 0 else "WEAK" if consistency >= 40 else "FAIL"
        r["rolling_avg_sharpe"] = round(avg_test, 4)
        r["rolling_consistency"] = round(consistency, 1)
        r["rolling_avg_trades_yr"] = round(avg_trades, 1)
        r["verdict"] = verdict
        r["n_windows"] = len(windows)

        tag = "***" if verdict == "PASS" else "   "
        print(f"  {tag} {r['strategy']:22}@{r['symbol']}: "
              f"OOS_Sh={avg_test:.4f} Cons={consistency:.0f}% "
              f"Tr/yr={avg_trades:.1f} [{verdict}] ({len(windows)} win)")
        if verdict == "PASS":
            validated.append(r)

    print(f"\n\n{'='*70}")
    print(f"VALIDATED (PASS): {len(validated)}")
    print("=" * 70)
    if validated:
        for r in validated:
            print(f"\n  {r['strategy']}@{r['symbol']}")
            print(f"    Val Sharpe:  {r['val_sharpe']:.4f}")
            print(f"    OOS Sharpe:  {r['rolling_avg_sharpe']:.4f}")
            print(f"    Consistency: {r['rolling_consistency']:.0f}%")
            print(f"    Trades/yr:   {r['rolling_avg_trades_yr']:.1f}")
            print(f"    Params:      {r['params']}")
    else:
        print("  No new strategies passed full validation.")

    Path("results").mkdir(exist_ok=True)
    if all_results:
        rows = []
        for r in all_results:
            flat = {k: v for k, v in r.items() if k != "params"}
            flat.update({f"p_{k}": v for k, v in r["params"].items()})
            rows.append(flat)
        pd.DataFrame(rows).to_csv("results/new_strategy_scan.csv", index=False)
        print(f"\nScan results -> results/new_strategy_scan.csv ({len(rows)} rows)")

    if validated:
        rows = []
        for r in validated:
            flat = {k: v for k, v in r.items() if k != "params"}
            flat.update({f"p_{k}": v for k, v in r["params"].items()})
            rows.append(flat)
        pd.DataFrame(rows).to_csv("results/new_strategy_validated.csv", index=False)
        print(f"Validated   -> results/new_strategy_validated.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
