"""
Direction scoring system backtest + full strategy walk-forward validation.

Phases:
  1. Direction prediction accuracy (hit rate, IC)
  2. Credit Spread: directional vs always-bull baseline
  3. Wheel CSP parameter scan (delta / DTE / TP%)
  4. ORB 0DTE backtest (synthetic 5min)
  5. Purged Walk-Forward with grid search
  6. Report generation

Usage:
  python run_direction_backtest.py                  # full run
  python run_direction_backtest.py --phase 1        # direction accuracy only
  python run_direction_backtest.py --phase 2        # credit spread only
  python run_direction_backtest.py --phase 5        # walk-forward only
"""
from __future__ import annotations

import sys
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from data.downloader import load_daily, _normalize_symbol
from data.indicators import TechnicalIndicators
from options.strategies.direction import DirectionAnalyzer, DEFAULT_PARAMS
from options.pricer import bs_price, compute_ivr
from options.backtest import _synth_iv, _hist_vol

# ── Config ──

CREDIT_SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
WHEEL_SYMBOLS = ["F", "AAL", "SOFI", "RIVN", "VALE", "PINS"]
ALL_SYMBOLS = CREDIT_SYMBOLS + WHEEL_SYMBOLS

FORWARD_DAYS = [5, 10, 21, 45]

REPORT_DIR = Path(__file__).resolve().parent / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "backtest_report.md"


# ═══════════════════════════════════════════════════════════════════
#  PHASE 1: Direction Prediction Accuracy
# ═══════════════════════════════════════════════════════════════════

def run_direction_accuracy(symbols: list[str] = CREDIT_SYMBOLS,
                           params: Optional[dict] = None) -> dict:
    """Score every day, compare with actual N-day forward return."""
    analyzer = DirectionAnalyzer(config=params)
    results = {}

    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 300:
            print(f"  [SKIP] {sym}: insufficient data")
            continue

        scored = analyzer.score_dataframe(df)
        valid = scored.dropna(subset=["dir_score"]).copy()
        if valid.empty:
            continue

        closes = scored["close"].values
        sym_result = {"symbol": sym, "total_signals": len(valid)}

        for n in FORWARD_DAYS:
            fwd_returns = []
            predictions = []
            scores_list = []

            for idx in valid.index:
                pos = scored.index.get_loc(idx)
                if pos + n >= len(closes):
                    break
                fwd_ret = (closes[pos + n] - closes[pos]) / closes[pos]
                score = valid.loc[idx, "dir_score"]
                direction = valid.loc[idx, "dir_direction"]

                fwd_returns.append(fwd_ret)
                scores_list.append(score)

                if direction == "BULL":
                    predictions.append(1)
                elif direction == "BEAR":
                    predictions.append(-1)
                else:
                    predictions.append(0)

            if len(fwd_returns) < 30:
                continue

            fwd_arr = np.array(fwd_returns)
            pred_arr = np.array(predictions)
            score_arr = np.array(scores_list)

            # Hit rate: BULL predicts positive return, BEAR predicts negative
            bull_mask = pred_arr == 1
            bear_mask = pred_arr == -1
            neutral_mask = pred_arr == 0

            bull_hit = np.mean(fwd_arr[bull_mask] > 0) * 100 if bull_mask.sum() > 0 else 0
            bear_hit = np.mean(fwd_arr[bear_mask] < 0) * 100 if bear_mask.sum() > 0 else 0
            overall_hit = 0
            if bull_mask.sum() + bear_mask.sum() > 0:
                correct = (bull_mask & (fwd_arr > 0)).sum() + (bear_mask & (fwd_arr < 0)).sum()
                overall_hit = correct / (bull_mask.sum() + bear_mask.sum()) * 100

            # IC: rank correlation of score vs forward return
            ic, ic_pval = sp_stats.spearmanr(score_arr, fwd_arr)

            # Average return conditional on signal
            bull_avg_ret = float(np.mean(fwd_arr[bull_mask])) * 100 if bull_mask.sum() > 0 else 0
            bear_avg_ret = float(np.mean(fwd_arr[bear_mask])) * 100 if bear_mask.sum() > 0 else 0
            neutral_avg_ret = float(np.mean(fwd_arr[neutral_mask])) * 100 if neutral_mask.sum() > 0 else 0

            # Baseline: always predict BULL
            baseline_hit = np.mean(fwd_arr > 0) * 100

            sym_result[f"{n}d"] = {
                "n_signals": len(fwd_returns),
                "n_bull": int(bull_mask.sum()),
                "n_bear": int(bear_mask.sum()),
                "n_neutral": int(neutral_mask.sum()),
                "bull_hit_pct": round(bull_hit, 1),
                "bear_hit_pct": round(bear_hit, 1),
                "overall_hit_pct": round(overall_hit, 1),
                "baseline_hit_pct": round(baseline_hit, 1),
                "ic": round(ic, 4),
                "ic_pval": round(ic_pval, 4),
                "bull_avg_ret_pct": round(bull_avg_ret, 2),
                "bear_avg_ret_pct": round(bear_avg_ret, 2),
                "neutral_avg_ret_pct": round(neutral_avg_ret, 2),
            }

        results[sym] = sym_result
        _print_sym_accuracy(sym, sym_result)

    return results


def _print_sym_accuracy(sym: str, r: dict):
    print(f"\n{'='*60}")
    print(f"  {sym} | 总信号数: {r['total_signals']}")
    print(f"{'='*60}")
    for n in FORWARD_DAYS:
        key = f"{n}d"
        if key not in r:
            continue
        d = r[key]
        edge = d["overall_hit_pct"] - d["baseline_hit_pct"]
        ic_star = "*" if d["ic_pval"] < 0.05 else ""
        print(f"  {n:>2}天 | 多:{d['n_bull']:>4} 空:{d['n_bear']:>4} 中性:{d['n_neutral']:>4} "
              f"| 命中:{d['overall_hit_pct']:>5.1f}% (基线:{d['baseline_hit_pct']:.1f}% "
              f"超额:{edge:+.1f}%) | IC={d['ic']:+.4f}{ic_star}")
        print(f"       | 多头信号平均收益:{d['bull_avg_ret_pct']:+.2f}% "
              f"空头信号:{d['bear_avg_ret_pct']:+.2f}% "
              f"中性:{d['neutral_avg_ret_pct']:+.2f}%")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 2: Credit Spread Directional vs Baseline
# ═══════════════════════════════════════════════════════════════════

def run_credit_spread_backtest(symbols: list[str] = CREDIT_SYMBOLS,
                               params: Optional[dict] = None,
                               spread_width: float = 5.0,
                               target_delta: float = 0.30,
                               max_hold: int = 21,
                               profit_take: float = 0.50,
                               stop_loss: float = 2.00,
                               min_ivr: float = 60.0,
                               risk_free: float = 0.05) -> dict:
    """Simulate Credit Spread with direction vs always-bull baseline."""
    analyzer = DirectionAnalyzer(config=params)
    results = {"directional": [], "baseline": []}

    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 300:
            continue

        scored = analyzer.score_dataframe(df)
        closes = scored["close"].values
        rets = np.diff(np.log(closes))

        for i in range(252, len(closes) - max_hold):
            # IVR filter
            window_vols = []
            for j in range(20, min(i, 252)):
                wv = float(np.std(rets[j - 20:j]) * np.sqrt(252))
                window_vols.append(wv)
            if not window_vols:
                continue
            current_vol = float(np.std(rets[i - 20:i]) * np.sqrt(252))
            ivr = compute_ivr(current_vol, window_vols)
            if ivr < min_ivr:
                continue

            spot = closes[i]
            iv = _synth_iv(current_vol, max_hold)
            T = max_hold / 252

            direction = scored["dir_direction"].iloc[i] if i < len(scored) else "NEUTRAL"
            score = scored["dir_score"].iloc[i] if i < len(scored) else 0

            # --- Directional trade ---
            if direction != "NEUTRAL" and not np.isnan(score):
                pnl_d = _sim_spread(
                    spot, iv, T, spread_width, target_delta, max_hold,
                    profit_take, stop_loss, risk_free, closes, i,
                    direction=direction,
                )
                results["directional"].append({
                    "symbol": sym, "date": scored["time_key"].iloc[i],
                    "direction": direction, "score": score, "pnl": pnl_d["pnl"],
                    "credit": pnl_d["credit"], "max_loss": pnl_d["max_loss"],
                    "reason": pnl_d["reason"],
                })

            # --- Baseline: always Bull Put ---
            pnl_b = _sim_spread(
                spot, iv, T, spread_width, target_delta, max_hold,
                profit_take, stop_loss, risk_free, closes, i,
                direction="BULL",
            )
            results["baseline"].append({
                "symbol": sym, "date": scored["time_key"].iloc[i],
                "direction": "BULL", "score": 0, "pnl": pnl_b["pnl"],
                "credit": pnl_b["credit"], "max_loss": pnl_b["max_loss"],
                "reason": pnl_b["reason"],
            })

    _print_credit_comparison(results)
    return results


def _sim_spread(spot, iv, T, width, target_delta, max_hold, tp_pct, sl_pct,
                r, closes, entry_idx, direction="BULL") -> dict:
    """Simulate a single credit spread trade."""
    if spot <= 0 or iv <= 0 or T <= 0:
        return {"pnl": 0, "credit": 0, "max_loss": 0, "reason": "invalid_input"}

    if direction == "BULL":
        short_strike = round(spot * (1 - target_delta * 0.5))
        long_strike = short_strike - width
        opt_type = "PUT"
    else:
        short_strike = round(spot * (1 + target_delta * 0.5))
        long_strike = short_strike + width
        opt_type = "CALL"

    if short_strike <= 0 or long_strike <= 0:
        return {"pnl": 0, "credit": 0, "max_loss": 0, "reason": "invalid_strike"}

    short_price = bs_price(spot, short_strike, T, r, iv, opt_type)
    long_price = bs_price(spot, long_strike, T, r, iv, opt_type)

    credit = short_price - long_price
    if credit <= 0.05:
        return {"pnl": 0, "credit": 0, "max_loss": 0, "reason": "no_credit"}

    max_loss = (abs(long_strike - short_strike) - credit) * 100
    pnl = credit * 100
    reason = "expiry"

    for d in range(1, min(max_hold + 1, len(closes) - entry_idx)):
        future_spot = closes[entry_idx + d]
        T_rem = max((max_hold - d) / 252, 0.001)
        short_now = bs_price(future_spot, short_strike, T_rem, r, iv, opt_type)
        long_now = bs_price(future_spot, long_strike, T_rem, r, iv, opt_type)
        spread_now = short_now - long_now
        cur_pnl = (credit - spread_now) * 100

        if cur_pnl >= credit * 100 * tp_pct:
            pnl = cur_pnl
            reason = "take_profit"
            break
        if cur_pnl <= -max_loss * sl_pct:
            pnl = cur_pnl
            reason = "stop_loss"
            break
        pnl = cur_pnl

    return {"pnl": pnl, "credit": credit, "max_loss": max_loss, "reason": reason}


def _print_credit_comparison(results: dict):
    print(f"\n{'='*70}")
    print("  Credit Spread: Directional vs Always-Bull Baseline")
    print(f"{'='*70}")

    for label in ["directional", "baseline"]:
        trades = results[label]
        if not trades:
            print(f"  [{label}] No trades")
            continue
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = sum(pnls)
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        avg_w = np.mean(wins) if wins else 0
        avg_l = np.mean(losses) if losses else 0
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(12)) if np.std(pnls) > 0 else 0
        pf = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else 999

        label_cn = "方向判断" if label == "directional" else "永远看涨"
        print(f"\n  [{label_cn}] 交易: {len(pnls)} | 总盈亏: ${total:+,.0f} "
              f"| 胜率: {wr:.1f}% | PF: {pf:.2f} | Sharpe: {sharpe:.2f}")
        print(f"    平均盈利: ${avg_w:+.0f} | 平均亏损: ${avg_l:+.0f}")

        if label == "directional":
            # Breakdown by direction
            bulls = [t for t in trades if t["direction"] == "BULL"]
            bears = [t for t in trades if t["direction"] == "BEAR"]
            for grp, grp_name in [(bulls, "BULL"), (bears, "BEAR")]:
                if grp:
                    g_pnls = [t["pnl"] for t in grp]
                    g_wr = sum(1 for p in g_pnls if p > 0) / len(g_pnls) * 100
                    print(f"    {grp_name}: {len(grp)}笔 | 盈亏: ${sum(g_pnls):+,.0f} "
                          f"| 胜率: {g_wr:.1f}%")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 3: Wheel CSP Parameter Scan
# ═══════════════════════════════════════════════════════════════════

def _precompute_ivr(closes: np.ndarray) -> np.ndarray:
    """Pre-compute rolling IVR for all bars (vectorized)."""
    n = len(closes)
    rets = np.diff(np.log(closes))
    ivr_arr = np.full(n, np.nan)

    # Rolling 20-day vol
    for i in range(252, n):
        current_vol = float(np.std(rets[i - 20:i]) * np.sqrt(252))
        # 1-year lookback for IVR percentile
        vols = []
        for j in range(max(20, i - 252), i, 5):  # step=5 for speed
            wv = float(np.std(rets[max(0, j - 20):j]) * np.sqrt(252))
            vols.append(wv)
        if vols:
            ivr_arr[i] = compute_ivr(current_vol, vols)
    return ivr_arr


def run_wheel_scan(symbols: list[str] = WHEEL_SYMBOLS,
                   risk_free: float = 0.05) -> dict:
    """Grid search over delta / DTE / TP% for Cash-Secured Put."""
    deltas = [0.20, 0.25, 0.30, 0.35]
    dtes = [21, 30, 45]
    tp_pcts = [0.50, 0.75, 1.00]

    results = []

    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 300:
            continue
        closes = df["close"].values
        print(f"  {sym}: pre-computing IVR...", end=" ", flush=True)
        ivr_arr = _precompute_ivr(closes)
        print("done")

        for delta, dte, tp in itertools.product(deltas, dtes, tp_pcts):
            trades = _sim_wheel_csp(closes, ivr_arr, delta, dte, tp, risk_free, min_ivr=30)
            if not trades:
                continue
            pnls = [t["pnl"] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(12)) if np.std(pnls) > 0 else 0

            results.append({
                "symbol": sym, "delta": delta, "dte": dte, "tp_pct": tp,
                "n_trades": len(trades), "total_pnl": total_pnl,
                "win_rate": wins / len(trades) * 100,
                "sharpe": sharpe,
                "avg_pnl": np.mean(pnls),
            })

    _print_wheel_results(results)
    return results


def _sim_wheel_csp(closes, ivr_arr, target_delta, dte, tp_pct, r, min_ivr=30):
    """Simulate selling CSP on daily data (pre-computed IVR)."""
    trades = []
    rets = np.diff(np.log(closes))
    i = 252
    while i < len(closes) - dte:
        ivr = ivr_arr[i] if i < len(ivr_arr) else np.nan
        if np.isnan(ivr) or ivr < min_ivr:
            i += 1
            continue

        spot = closes[i]
        if spot <= 0:
            i += 1
            continue
        current_vol = float(np.std(rets[max(0, i - 20):i]) * np.sqrt(252))
        strike = round(spot * (1 - target_delta * 0.5), 1)
        if strike <= 0:
            i += 1
            continue
        T = dte / 252
        iv = _synth_iv(current_vol, dte)
        put_price = bs_price(spot, strike, T, r, iv, "PUT")

        if put_price < 0.05:
            i += 1
            continue

        credit = put_price * 100
        pnl = credit
        reason = "expiry"

        for d in range(1, min(dte + 1, len(closes) - i)):
            future = closes[i + d]
            T_rem = max((dte - d) / 252, 0.001)
            put_now = bs_price(future, strike, T_rem, r, iv, "PUT")
            cur_pnl = (put_price - put_now) * 100

            if cur_pnl >= credit * tp_pct:
                pnl = cur_pnl
                reason = "take_profit"
                break
            pnl = cur_pnl

        if reason == "expiry" and i + dte < len(closes):
            final = closes[i + dte]
            if final < strike:
                pnl = (strike - final + put_price) * -100 + credit
                reason = "assigned"
            else:
                pnl = credit

        trades.append({"pnl": pnl, "reason": reason})
        i += dte + 1

    return trades


def _print_wheel_results(results: list[dict]):
    print(f"\n{'='*70}")
    print("  Wheel CSP Parameter Scan Results")
    print(f"{'='*70}")

    if not results:
        print("  No results.")
        return

    rdf = pd.DataFrame(results)

    # Best per symbol
    for sym in rdf["symbol"].unique():
        sym_df = rdf[rdf["symbol"] == sym].sort_values("sharpe", ascending=False)
        best = sym_df.iloc[0]
        print(f"\n  {sym} 最优: delta={best['delta']:.2f} DTE={best['dte']} "
              f"TP={best['tp_pct']:.0%} | {best['n_trades']}笔 "
              f"| 总盈亏=${best['total_pnl']:+,.0f} | 胜率={best['win_rate']:.1f}% "
              f"| Sharpe={best['sharpe']:.2f}")

    # Overall best combo
    overall = rdf.groupby(["delta", "dte", "tp_pct"]).agg(
        avg_sharpe=("sharpe", "mean"), total_trades=("n_trades", "sum"),
        avg_wr=("win_rate", "mean"),
    ).sort_values("avg_sharpe", ascending=False)
    print(f"\n  全标的最优参数组合:")
    for idx, row in overall.head(5).iterrows():
        print(f"    delta={idx[0]:.2f} DTE={idx[1]} TP={idx[2]:.0%} "
              f"| avg Sharpe={row['avg_sharpe']:.2f} | avg WR={row['avg_wr']:.1f}%")


# ═══════════════════════════════════════════════════════════════════
#  PHASE 4: ORB 0DTE Backtest (synthetic 5min)
# ═══════════════════════════════════════════════════════════════════

def run_orb_backtest(symbols: list[str] = ["SPY", "QQQ"]) -> dict:
    """ORB 0DTE backtest using synthesized 5min data."""
    from data.synthesizer import load_or_synthesize_5min
    from options.backtest import OptionsBacktester

    bt = OptionsBacktester()
    results = {}

    for sym in symbols:
        df_5min = load_or_synthesize_5min(sym, min_days=200)
        if df_5min is None or len(df_5min) < 500:
            print(f"  [SKIP] {sym}: insufficient 5min data")
            continue

        config = {
            "stop_loss_pct": 0.50,
            "target_pct": 1.00,
            "max_premium": 200,
        }
        result = bt.backtest_orb(df_5min, config)
        results[sym] = {
            "total_trades": result.total_trades,
            "total_pnl": result.total_pnl,
            "win_rate": result.win_rate,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
        }

        print(f"\n  {sym} ORB 0DTE: {result.total_trades}笔 | "
              f"盈亏=${result.total_pnl:+,.0f} | 胜率={result.win_rate:.1f}% "
              f"| Sharpe={result.sharpe:.2f} | MaxDD=${result.max_drawdown:+,.0f}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  PHASE 5: Purged Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════

PARAM_GRID = {
    "trend_weight": [0.30, 0.40, 0.50],
    "momentum_weight": [0.15, 0.25, 0.35],
    "macd_weight": [0.15, 0.20, 0.25],
    "min_score": [15, 20, 25, 30, 35],
    "rsi_overbought": [65, 70, 75],
    "rsi_oversold": [25, 30, 35],
}


def _rolling_wf_splits(n: int, train_years: int = 3, test_years: int = 1,
                        bars_per_year: int = 252, purge_bars: int = 5):
    """Generate rolling walk-forward (train, test) index tuples."""
    train_len = train_years * bars_per_year
    test_len = test_years * bars_per_year
    splits = []
    start = 0
    while start + train_len + purge_bars + test_len <= n:
        train_end = start + train_len
        test_start = train_end + purge_bars
        test_end = min(test_start + test_len, n)
        if test_end <= test_start:
            break
        splits.append((start, train_end, test_start, test_end))
        start += test_len  # roll forward by test_len
    return splits


def _precompute_slice(df_slice: pd.DataFrame) -> dict:
    """Pre-compute indicators and IVR once for a data slice."""
    df = df_slice.copy()
    df = TechnicalIndicators.add_ma(df, period=20)
    df = TechnicalIndicators.add_ma(df, period=50)
    df = TechnicalIndicators.add_rsi(df, period=14)
    df = TechnicalIndicators.add_macd(df)
    df = TechnicalIndicators.add_atr(df, period=14)

    closes = df["close"].values
    rets = np.diff(np.log(closes))

    ivr_arr = np.full(len(closes), np.nan)
    vol_arr = np.full(len(closes), np.nan)
    for i in range(252, len(closes)):
        current_vol = float(np.std(rets[i - 20:i]) * np.sqrt(252))
        vol_arr[i] = current_vol
        vols = []
        for j in range(max(20, i - 252), i, 5):
            wv = float(np.std(rets[max(0, j - 20):j]) * np.sqrt(252))
            vols.append(wv)
        if vols:
            ivr_arr[i] = compute_ivr(current_vol, vols)

    return {"df": df, "closes": closes, "ivr": ivr_arr, "vol": vol_arr}


def _fast_score_with_params(precomp: dict, params: dict,
                             spread_width: float = 5.0, max_hold: int = 21) -> dict:
    """Score pre-computed data with specific params and simulate trades."""
    df = precomp["df"]
    closes = precomp["closes"]
    ivr_arr = precomp["ivr"]
    vol_arr = precomp["vol"]

    analyzer = DirectionAnalyzer(config=params)
    trades = []

    for i in range(max(252, 55), len(closes) - max_hold):
        if np.isnan(ivr_arr[i]) or ivr_arr[i] < 60:
            continue
        if np.isnan(vol_arr[i]):
            continue

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        if pd.isna(row.get("ma_20")) or pd.isna(row.get("ma_50")) or pd.isna(row.get("rsi_14")):
            continue

        atr_window = df["atr_14"].iloc[max(0, i - 19):i + 1].dropna()
        atr_avg = float(atr_window.mean()) if len(atr_window) > 0 else float(row["atr_14"])

        sig = analyzer._compute_score(
            price=float(row["close"]), sma20=float(row["ma_20"]),
            sma50=float(row["ma_50"]), rsi=float(row["rsi_14"]),
            macd_hist=float(row["macd_hist"]),
            macd_hist_prev=float(prev_row["macd_hist"]),
            atr=float(row["atr_14"]), atr_avg=atr_avg,
        )

        if sig.direction == "NEUTRAL":
            continue

        spot = closes[i]
        iv = _synth_iv(vol_arr[i], max_hold)
        T = max_hold / 252

        result = _sim_spread(spot, iv, T, spread_width, 0.30, max_hold,
                             0.50, 2.00, 0.05, closes, i, direction=sig.direction)
        if result["credit"] > 0:
            trades.append(result["pnl"])

    if len(trades) < 5:
        return {"sharpe": -999, "total_pnl": 0, "n_trades": 0, "win_rate": 0}

    pnls = np.array(trades)
    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(12)) if np.std(pnls) > 0 else 0
    wr = np.mean(pnls > 0) * 100

    return {
        "sharpe": sharpe,
        "total_pnl": float(np.sum(pnls)),
        "n_trades": len(trades),
        "win_rate": float(wr),
    }


def _grid_search_on_slice(df_slice: pd.DataFrame, param_grid: dict,
                           top_n: int = 10) -> list[dict]:
    """Grid search with pre-computed indicators (much faster)."""
    precomp = _precompute_slice(df_slice)
    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        metrics = _fast_score_with_params(precomp, params)
        results.append({"params": params, **metrics})

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results[:top_n]


def run_walk_forward(symbols: list[str] = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]) -> dict:
    """Purged walk-forward validation across multiple symbols."""
    print(f"\n{'='*70}")
    print("  Walk-Forward Validation (3yr train + 1yr test)")
    print(f"{'='*70}")

    # Merge all symbol data for a portfolio-level WF
    all_results = []

    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 1200:
            print(f"  [SKIP] {sym}: need >= 1200 bars for WF")
            continue

        n = len(df)
        splits = _rolling_wf_splits(n)
        if not splits:
            print(f"  [SKIP] {sym}: not enough data for WF splits")
            continue

        print(f"\n  {sym}: {len(splits)} WF windows")
        sym_windows = []

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
            train_df = df.iloc[train_start:train_end].reset_index(drop=True)
            # Include lookback before test for IVR warmup (need 252+ bars)
            lookback = 260
            test_expanded_start = max(0, test_start - lookback)
            test_with_lookback = df.iloc[test_expanded_start:test_end].reset_index(drop=True)

            train_dates = f"{train_df['time_key'].iloc[0]} -> {train_df['time_key'].iloc[-1]}"
            test_dates = f"{df['time_key'].iloc[test_start]} -> {df['time_key'].iloc[min(test_end - 1, len(df) - 1)]}"

            print(f"    Window {w_idx+1}: train({train_dates}) test({test_dates})")

            reduced_grid = {
                "trend_weight": [0.30, 0.40, 0.50],
                "momentum_weight": [0.20, 0.30],
                "macd_weight": [0.15, 0.25],
                "min_score": [15, 25, 35],
                "rsi_overbought": [65, 75],
                "rsi_oversold": [25, 35],
            }
            top_train = _grid_search_on_slice(train_df, reduced_grid, top_n=5)

            if not top_train or top_train[0]["sharpe"] <= -999:
                print(f"      No viable params on train set")
                sym_windows.append({"train_sharpe": 0, "test_sharpe": 0, "params": {}})
                continue

            best_params = top_train[0]["params"]
            train_sharpe = top_train[0]["sharpe"]

            test_precomp = _precompute_slice(test_with_lookback)
            test_metrics = _fast_score_with_params(test_precomp, best_params)
            test_sharpe = test_metrics["sharpe"]

            overfit = (1 - test_sharpe / train_sharpe) * 100 if train_sharpe > 0 else 999
            print(f"      Best: {best_params}")
            print(f"      Train: Sharpe={train_sharpe:.2f} ({top_train[0]['n_trades']}笔)")
            print(f"      Test:  Sharpe={test_sharpe:.2f} ({test_metrics['n_trades']}笔) "
                  f"Overfit={overfit:+.0f}%")

            sym_windows.append({
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "overfit_pct": overfit,
                "params": best_params,
                "test_n_trades": test_metrics["n_trades"],
                "test_win_rate": test_metrics["win_rate"],
            })

        # WF consistency
        test_sharpes = [w["test_sharpe"] for w in sym_windows if w.get("test_sharpe", -999) > -999]
        consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100 if test_sharpes else 0
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0

        print(f"    {sym} WF 一致性: {consistency:.0f}% (Sharpe>0的窗口比例)")
        print(f"    {sym} 平均 OOS Sharpe: {avg_test_sharpe:.2f}")

        all_results.append({
            "symbol": sym,
            "n_windows": len(sym_windows),
            "consistency_pct": consistency,
            "avg_test_sharpe": float(avg_test_sharpe),
            "windows": sym_windows,
        })

    return {"symbols": all_results}


# ═══════════════════════════════════════════════════════════════════
#  PHASE 6: Report Generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(direction_results: dict, credit_results: dict,
                    wheel_results, orb_results: dict,
                    wf_results: dict):
    """Generate comprehensive markdown report."""
    lines = [
        "# 方向评分系统回测 + 全策略 Walk-Forward 验证报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\n数据范围: 2016-05 ~ 2026-04 (约10年日线)",
        "",
    ]

    # Section 1: Direction Accuracy
    lines.append("## 1. 方向评分系统准确率")
    lines.append("")
    lines.append("| 标的 | 前瞻天数 | BULL数 | BEAR数 | 命中率 | 基线 | 超额 | IC |")
    lines.append("|------|---------|--------|--------|--------|------|------|-----|")

    for sym, r in direction_results.items():
        for n in FORWARD_DAYS:
            key = f"{n}d"
            if key not in r:
                continue
            d = r[key]
            edge = d["overall_hit_pct"] - d["baseline_hit_pct"]
            ic_str = f"{d['ic']:+.4f}" + ("*" if d["ic_pval"] < 0.05 else "")
            lines.append(f"| {sym} | {n}天 | {d['n_bull']} | {d['n_bear']} | "
                        f"{d['overall_hit_pct']:.1f}% | {d['baseline_hit_pct']:.1f}% | "
                        f"{edge:+.1f}% | {ic_str} |")

    # Section 2: Credit Spread Comparison
    lines.append("")
    lines.append("## 2. Credit Spread: 方向判断 vs 永远看涨")
    lines.append("")

    for label in ["directional", "baseline"]:
        trades = credit_results.get(label, [])
        if not trades:
            continue
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        wr = wins / len(pnls) * 100 if pnls else 0
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(12)) if np.std(pnls) > 0 else 0
        label_cn = "方向判断" if label == "directional" else "永远看涨(基线)"
        lines.append(f"**{label_cn}**: {len(pnls)}笔交易 | 总盈亏 ${total:+,.0f} | "
                    f"胜率 {wr:.1f}% | Sharpe {sharpe:.2f}")

    # Section 3: Wheel CSP
    lines.append("")
    lines.append("## 3. Wheel CSP 参数扫描")
    lines.append("")

    if isinstance(wheel_results, list) and wheel_results:
        wdf = pd.DataFrame(wheel_results)
        overall = wdf.groupby(["delta", "dte", "tp_pct"]).agg(
            avg_sharpe=("sharpe", "mean"), avg_wr=("win_rate", "mean"),
        ).sort_values("avg_sharpe", ascending=False)
        lines.append("| Delta | DTE | 止盈 | Avg Sharpe | Avg WinRate |")
        lines.append("|-------|-----|------|------------|-------------|")
        for idx, row in overall.head(10).iterrows():
            lines.append(f"| {idx[0]:.2f} | {idx[1]} | {idx[2]:.0%} | "
                        f"{row['avg_sharpe']:.2f} | {row['avg_wr']:.1f}% |")

    # Section 4: ORB
    lines.append("")
    lines.append("## 4. ORB 0DTE 回测")
    lines.append("")
    for sym, r in orb_results.items():
        lines.append(f"**{sym}**: {r['total_trades']}笔 | 盈亏 ${r['total_pnl']:+,.0f} | "
                    f"胜率 {r['win_rate']:.1f}% | Sharpe {r['sharpe']:.2f}")

    # Section 5: Walk-Forward
    lines.append("")
    lines.append("## 5. Walk-Forward 验证")
    lines.append("")
    lines.append("| 标的 | 窗口数 | OOS一致性 | 平均OOS Sharpe |")
    lines.append("|------|--------|-----------|---------------|")

    for sr in wf_results.get("symbols", []):
        lines.append(f"| {sr['symbol']} | {sr['n_windows']} | "
                    f"{sr['consistency_pct']:.0f}% | {sr['avg_test_sharpe']:.2f} |")

    # Section 6: Recommendations
    lines.append("")
    lines.append("## 6. 推荐参数")
    lines.append("")

    # Find best WF params
    best_params_count: dict = {}
    for sr in wf_results.get("symbols", []):
        for w in sr.get("windows", []):
            p = w.get("params", {})
            if not p:
                continue
            key = json.dumps(p, sort_keys=True)
            best_params_count[key] = best_params_count.get(key, 0) + 1

    if best_params_count:
        most_common = max(best_params_count, key=best_params_count.get)
        rec_params = json.loads(most_common)
        lines.append("Walk-Forward 中出现频率最高的最优参数组合:")
        lines.append("```yaml")
        lines.append("direction:")
        for k, v in sorted(rec_params.items()):
            lines.append(f"  {k}: {v}")
        lines.append("```")

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\n报告已生成: {REPORT_PATH}")
    return report_text


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Direction Backtest + Walk-Forward")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1-6), 0=all")
    args = parser.parse_args()

    print("=" * 70)
    print("  方向评分系统回测 + 全策略 Walk-Forward 验证")
    print("=" * 70)

    direction_results = {}
    credit_results = {}
    wheel_results = []
    orb_results = {}
    wf_results = {}

    if args.phase in (0, 1):
        print("\n\n══ PHASE 1: 方向预测准确率 ══")
        direction_results = run_direction_accuracy()

    if args.phase in (0, 2):
        print("\n\n══ PHASE 2: Credit Spread 方向 vs 基线 ══")
        credit_results = run_credit_spread_backtest()

    if args.phase in (0, 3):
        print("\n\n══ PHASE 3: Wheel CSP 参数扫描 ══")
        wheel_results = run_wheel_scan()

    if args.phase in (0, 4):
        print("\n\n══ PHASE 4: ORB 0DTE 回测 ══")
        orb_results = run_orb_backtest()

    if args.phase in (0, 5):
        print("\n\n══ PHASE 5: Walk-Forward 验证 ══")
        wf_results = run_walk_forward()

    if args.phase in (0, 6):
        print("\n\n══ PHASE 6: 生成报告 ══")
        generate_report(direction_results, credit_results,
                       wheel_results, orb_results, wf_results)


if __name__ == "__main__":
    main()
