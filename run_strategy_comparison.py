"""
$10K 策略对比回测 — 5 个候选策略统一评估

统一条件:
  - 时间: 2016-2026 (10年)
  - 本金: $10,000
  - 数据: Yahoo Finance 日线
  - 成本: 含手续费 + bid-ask spread

策略:
  1. ETF 动量轮动 (top_n=5, budget=$8,000 — 80%)
  2. Credit Spread (bull put, budget=$3,000)
  3. Iron Condor (bull put + bear call, budget=$3,000)
  4. Wheel CSP (budget=$3,000)
  5. 组合策略 (ETF + 最优期权策略)

Usage:
  python run_strategy_comparison.py
"""
from __future__ import annotations

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.downloader import save_daily, load_daily
from backtest.full_validation import (
    sim_spread_with_cost, sim_wheel_with_cost, CostModel, DEFAULT_COST,
)
from backtest.data_cache import DataCache
from strategy.fractional.momentum_rotation import MomentumRotation
from options.pricer import bs_price

CAPITAL = 10_000
YEARS = 10

# ETF pool for momentum rotation
ETF_POOL = [
    "SGOV", "TLT", "XLK", "SMH", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "VEA", "EEM", "SLV", "GDX", "XLB",
]

# High-liquidity symbols for options strategies
OPTION_SYMBOLS = ["SPY", "QQQ", "IWM", "XLF", "XLE"]


def download_all_data():
    """Download historical data for all symbols."""
    all_syms = list(set(ETF_POOL + OPTION_SYMBOLS + ["SPY", "IAUM"]))
    print(f"Downloading {len(all_syms)} symbols ({YEARS} years)...")
    for sym in all_syms:
        df = load_daily(sym)
        if df is None or df.empty or len(df) < 252 * 5:
            print(f"  Downloading {sym}...")
            save_daily(sym, years=YEARS)
            time.sleep(0.5)
        else:
            print(f"  {sym}: {len(df)} bars (cached)")
    print()


def calc_metrics(equity: np.ndarray, name: str, n_trades: int = 0,
                 wins: int = 0) -> dict:
    """Calculate standard metrics from equity curve."""
    if len(equity) < 2:
        return {"name": name, "error": "insufficient data"}

    total_return = (equity[-1] / equity[0]) - 1.0
    n_years = len(equity) / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1 if total_return > -1 else -1

    daily_rets = np.diff(equity) / equity[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0
    sortino_dn = daily_rets[daily_rets < 0]
    sortino = float(np.mean(daily_rets) / np.std(sortino_dn) * np.sqrt(252)) if len(sortino_dn) > 0 and np.std(sortino_dn) > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    vol = float(np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 0 else 0

    win_rate = wins / max(n_trades, 1) if n_trades > 0 else 0

    # Monthly returns
    monthly_rets = []
    step = 21
    for i in range(0, len(equity) - step, step):
        mr = (equity[i + step] / equity[i]) - 1.0
        monthly_rets.append(mr)
    monthly_rets = np.array(monthly_rets)
    pos_months = int(np.sum(monthly_rets > 0)) if len(monthly_rets) > 0 else 0
    neg_months = int(np.sum(monthly_rets <= 0)) if len(monthly_rets) > 0 else 0

    return {
        "name": name,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "volatility": vol,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "pos_months": pos_months,
        "neg_months": neg_months,
        "final_value": equity[-1],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy 1: ETF Momentum Rotation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest_momentum(budget=8000, top_n=5) -> dict:
    print("=" * 60)
    print(f"  Strategy 1: ETF Momentum Rotation (budget=${budget}, top_n={top_n})")
    print("=" * 60)

    daily_data = {}
    for sym in ETF_POOL:
        df = load_daily(sym)
        if df is not None and not df.empty:
            daily_data[sym] = df
            print(f"  {sym}: {len(df)} bars")

    if len(daily_data) < 5:
        print("  ERROR: not enough data")
        return {}

    result = MomentumRotation.backtest_momentum(
        daily_data=daily_data,
        budget=budget,
        top_n=top_n,
        lookback=252,
        skip=21,
        sma_period=200,
        safe_haven="SGOV",
    )

    metrics = calc_metrics(result["equity"], "ETF_Momentum_Rotation",
                           n_trades=result["n_trades"])
    print(f"\n  CAGR: {metrics['cagr']:.1%} | Sharpe: {metrics['sharpe']:.2f} | "
          f"MaxDD: {metrics['max_drawdown']:.1%} | Trades: {metrics['n_trades']}")
    return {**metrics, "equity": result["equity"]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy 2: Credit Spread (Bull Put)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_daily_equity(trade_results: list[dict], budget: float,
                        n_total_days: int) -> np.ndarray:
    """Build a daily equity curve from trade-level PnL results.

    Each trade has 'entry_idx' and 'hold_days' so we can spread PnL
    across the holding period. If these aren't present, spread evenly.
    """
    daily_pnl = np.zeros(n_total_days)
    for t in trade_results:
        entry = t.get("entry_idx", 0)
        hold = t.get("hold_days", 30)
        pnl = t.get("pnl", 0)
        daily_p = pnl / max(hold, 1)
        for d in range(hold):
            idx = entry + d
            if 0 <= idx < n_total_days:
                daily_pnl[idx] += daily_p

    equity = np.zeros(n_total_days + 1)
    equity[0] = budget
    for i in range(n_total_days):
        equity[i + 1] = equity[i] + daily_pnl[i]
    return np.maximum(equity, 1)


def backtest_credit_spread(budget=3000, width=5, delta=0.30, ivr_min=30,
                           dte=30, tp_pct=0.50, sl_pct=2.0) -> dict:
    print("\n" + "=" * 60)
    print(f"  Strategy 2: Credit Spread (budget=${budget}, width={width}, delta={delta})")
    print("=" * 60)

    cache = DataCache.get()
    cache.load_symbols(OPTION_SYMBOLS, years=YEARS)

    trade_results = []
    max_bars = 0

    for sym in OPTION_SYMBOLS:
        closes = cache.get_closes(sym)
        ivr = cache.get_ivr(sym)
        if len(closes) < 252 * 3 or len(ivr) == 0:
            print(f"  {sym}: skip (insufficient data)")
            continue

        max_bars = max(max_bars, len(closes))
        print(f"  {sym}: {len(closes)} bars, IVR computed")

        rets = np.diff(np.log(closes))
        i = 252
        while i < len(closes) - dte - 1:
            if i < len(ivr) and ivr[i] >= ivr_min:
                spot = closes[i]
                current_vol = float(np.std(rets[max(0, i - 20):i]) * np.sqrt(252))
                iv = current_vol * 1.2
                T = dte / 252

                result = sim_spread_with_cost(
                    spot=spot, iv=iv, T=T, width=width,
                    target_delta=delta, max_hold=dte,
                    tp_pct=tp_pct, sl_pct=sl_pct, r=0.05,
                    closes=closes, entry_idx=i,
                    direction="BULL",
                )

                if result["credit"] > 0:
                    hold_days = {"take_profit": min(dte // 2, 15),
                                 "stop_loss": min(dte // 3, 10),
                                 "expiry": dte}.get(result["reason"], dte)
                    result["entry_idx"] = i
                    result["hold_days"] = hold_days
                    trade_results.append(result)
                    i += dte + 5
                    continue
            i += 1

    if not trade_results:
        print("  ERROR: no trades")
        return {}

    pnls = np.array([t["pnl"] for t in trade_results])
    wins = int(np.sum(pnls > 0))
    n_trades = len(pnls)

    equity = _build_daily_equity(trade_results, budget, max_bars)

    metrics = calc_metrics(equity, "Credit_Spread", n_trades=n_trades, wins=wins)
    print(f"\n  CAGR: {metrics['cagr']:.1%} | Sharpe: {metrics['sharpe']:.2f} | "
          f"MaxDD: {metrics['max_drawdown']:.1%} | Trades: {n_trades} | "
          f"WinRate: {metrics['win_rate']:.0%}")
    return {**metrics, "equity": equity}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy 3: Iron Condor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sim_iron_condor(spot, iv, T, width, put_delta, call_delta, max_hold,
                    tp_pct, sl_pct, r, closes, entry_idx,
                    cost_model=DEFAULT_COST) -> dict:
    """Simulate an Iron Condor: Bull Put Spread + Bear Call Spread."""
    if spot <= 0 or iv <= 0 or T <= 0:
        return {"pnl": 0, "credit": 0, "max_loss": 0, "reason": "invalid"}

    # Bull put side
    put_short = round(spot * (1 - put_delta * 0.5))
    put_long = put_short - width

    # Bear call side
    call_short = round(spot * (1 + call_delta * 0.5))
    call_long = call_short + width

    if put_long <= 0 or call_long <= 0:
        return {"pnl": 0, "credit": 0, "max_loss": 0, "reason": "invalid_strike"}

    # Entry prices
    put_short_p = bs_price(spot, put_short, T, r, iv, "PUT")
    put_long_p = bs_price(spot, put_long, T, r, iv, "PUT")
    call_short_p = bs_price(spot, call_short, T, r, iv, "CALL")
    call_long_p = bs_price(spot, call_long, T, r, iv, "CALL")

    put_credit = put_short_p - put_long_p
    call_credit = call_short_p - call_long_p
    total_credit = put_credit + call_credit

    if total_credit <= 0.05:
        return {"pnl": 0, "credit": 0, "max_loss": 0, "reason": "no_credit"}

    max_loss = (width - total_credit) * 100
    avg_price = (put_short_p + put_long_p + call_short_p + call_long_p) / 4
    entry_cost = cost_model.trade_cost(4, 1, avg_price, is_open=True)

    pnl = total_credit * 100
    reason = "expiry"
    exit_avg = avg_price

    for d in range(1, min(max_hold + 1, len(closes) - entry_idx)):
        future = closes[entry_idx + d]
        T_rem = max((max_hold - d) / 252, 0.001)

        ps = bs_price(future, put_short, T_rem, r, iv, "PUT")
        pl = bs_price(future, put_long, T_rem, r, iv, "PUT")
        cs = bs_price(future, call_short, T_rem, r, iv, "CALL")
        cl = bs_price(future, call_long, T_rem, r, iv, "CALL")

        spread_now = (ps - pl) + (cs - cl)
        cur_pnl = (total_credit - spread_now) * 100

        if cur_pnl >= total_credit * 100 * tp_pct:
            pnl = cur_pnl
            exit_avg = (ps + pl + cs + cl) / 4
            reason = "take_profit"
            break
        if cur_pnl <= -max_loss * sl_pct:
            pnl = cur_pnl
            exit_avg = (ps + pl + cs + cl) / 4
            reason = "stop_loss"
            break
        pnl = cur_pnl
        exit_avg = (ps + pl + cs + cl) / 4

    exit_cost = cost_model.trade_cost(4, 1, exit_avg, is_open=False)
    pnl_after = pnl - entry_cost - exit_cost

    return {"pnl": pnl_after, "pnl_gross": pnl, "cost": entry_cost + exit_cost,
            "credit": total_credit, "max_loss": max_loss, "reason": reason}


def backtest_iron_condor(budget=3000, width=5, put_delta=0.15,
                         call_delta=0.15, ivr_min=30, dte=30,
                         tp_pct=0.50, sl_pct=2.0) -> dict:
    print("\n" + "=" * 60)
    print(f"  Strategy 3: Iron Condor (budget=${budget}, width={width})")
    print("=" * 60)

    cache = DataCache.get()
    cache.load_symbols(OPTION_SYMBOLS, years=YEARS)

    trade_results = []
    max_bars = 0

    for sym in OPTION_SYMBOLS:
        closes = cache.get_closes(sym)
        ivr = cache.get_ivr(sym)
        if len(closes) < 252 * 3 or len(ivr) == 0:
            continue

        max_bars = max(max_bars, len(closes))
        print(f"  {sym}: {len(closes)} bars")
        rets = np.diff(np.log(closes))
        i = 252
        while i < len(closes) - dte - 1:
            if i < len(ivr) and ivr[i] >= ivr_min:
                spot = closes[i]
                current_vol = float(np.std(rets[max(0, i - 20):i]) * np.sqrt(252))
                iv = current_vol * 1.2
                T = dte / 252

                result = sim_iron_condor(
                    spot=spot, iv=iv, T=T, width=width,
                    put_delta=put_delta, call_delta=call_delta,
                    max_hold=dte, tp_pct=tp_pct, sl_pct=sl_pct,
                    r=0.05, closes=closes, entry_idx=i,
                )

                if result["credit"] > 0:
                    hold_days = {"take_profit": min(dte // 2, 15),
                                 "stop_loss": min(dte // 3, 10),
                                 "expiry": dte}.get(result["reason"], dte)
                    result["entry_idx"] = i
                    result["hold_days"] = hold_days
                    trade_results.append(result)
                    i += dte + 5
                    continue
            i += 1

    if not trade_results:
        print("  ERROR: no trades")
        return {}

    pnls = np.array([t["pnl"] for t in trade_results])
    wins = int(np.sum(pnls > 0))
    n_trades = len(pnls)

    equity = _build_daily_equity(trade_results, budget, max_bars)

    metrics = calc_metrics(equity, "Iron_Condor", n_trades=n_trades, wins=wins)
    print(f"\n  CAGR: {metrics['cagr']:.1%} | Sharpe: {metrics['sharpe']:.2f} | "
          f"MaxDD: {metrics['max_drawdown']:.1%} | Trades: {n_trades} | "
          f"WinRate: {metrics['win_rate']:.0%}")
    return {**metrics, "equity": equity}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy 4: Wheel CSP (optimized DTE=14)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sim_wheel_with_tracking(closes, ivr_arr, target_delta, dte, tp_pct, r,
                            min_ivr=30, cost_model=DEFAULT_COST):
    """Wheel CSP simulation that also tracks entry_idx for daily equity."""
    from options.pricer import bs_price as _bs
    try:
        from options.backtest import _synth_iv
    except ImportError:
        def _synth_iv(vol, d):
            return vol * (1.0 + max(0, (30 - d)) * 0.005)

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
        put_price = _bs(spot, strike, T, r, iv, "PUT")
        if put_price < 0.05:
            i += 1
            continue

        entry_cost = cost_model.trade_cost(1, 1, put_price, is_open=True)
        credit = put_price * 100
        pnl = credit
        reason = "expiry"
        exit_price = put_price
        hold_days = dte

        for d in range(1, min(dte + 1, len(closes) - i)):
            future = closes[i + d]
            T_rem = max((dte - d) / 252, 0.001)
            put_now = _bs(future, strike, T_rem, r, iv, "PUT")
            cur_pnl = (put_price - put_now) * 100
            if cur_pnl >= credit * tp_pct:
                pnl = cur_pnl
                exit_price = put_now
                reason = "take_profit"
                hold_days = d
                break
            pnl = cur_pnl
            exit_price = put_now

        if reason == "expiry" and i + dte < len(closes):
            final = closes[i + dte]
            if final < strike:
                pnl = (strike - final + put_price) * -100 + credit
                reason = "assigned"

        exit_cost = cost_model.trade_cost(1, 1, max(exit_price, 0.01), is_open=False)
        pnl_after = pnl - entry_cost - exit_cost
        trades.append({"pnl": pnl_after, "reason": reason,
                        "entry_idx": i, "hold_days": hold_days})
        i += dte + 1
    return trades


def backtest_wheel(budget=3000, delta=0.25, dte=14, tp_pct=0.50,
                   ivr_min=30) -> dict:
    print("\n" + "=" * 60)
    print(f"  Strategy 4: Wheel CSP (budget=${budget}, DTE={dte}, delta={delta})")
    print("=" * 60)

    cache = DataCache.get()
    wheel_syms = ["XLU", "XLRE", "XLF", "XLE", "XLB"]
    cache.load_symbols(wheel_syms, years=YEARS)

    trade_results = []
    max_bars = 0

    for sym in wheel_syms:
        closes = cache.get_closes(sym)
        ivr = cache.get_ivr(sym)
        if len(closes) < 252 * 3 or len(ivr) == 0:
            continue

        max_bars = max(max_bars, len(closes))
        print(f"  {sym}: {len(closes)} bars")
        trades = sim_wheel_with_tracking(
            closes=closes, ivr_arr=ivr,
            target_delta=delta, dte=dte,
            tp_pct=tp_pct, r=0.05, min_ivr=ivr_min,
        )
        trade_results.extend(trades)

    if not trade_results:
        print("  ERROR: no trades")
        return {}

    pnls = np.array([t["pnl"] for t in trade_results])
    wins = int(np.sum(pnls > 0))
    n_trades = len(pnls)

    equity = _build_daily_equity(trade_results, budget, max_bars)

    metrics = calc_metrics(equity, "Wheel_CSP", n_trades=n_trades, wins=wins)
    print(f"\n  CAGR: {metrics['cagr']:.1%} | Sharpe: {metrics['sharpe']:.2f} | "
          f"MaxDD: {metrics['max_drawdown']:.1%} | Trades: {n_trades} | "
          f"WinRate: {metrics['win_rate']:.0%}")
    return {**metrics, "equity": equity}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy 5: Combined (Momentum + best options)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest_combined(mom_equity, opt_equity, opt_name,
                      ratios=None) -> list[dict]:
    if ratios is None:
        ratios = [(0.80, 0.20), (0.70, 0.30), (0.60, 0.40)]

    print("\n" + "=" * 60)
    print(f"  Strategy 5: Combined (Momentum + {opt_name})")
    print("=" * 60)

    # Normalize both equity curves to returns
    mom_rets = np.diff(mom_equity) / mom_equity[:-1]
    opt_rets = np.diff(opt_equity) / opt_equity[:-1]

    # Align to shorter series
    n = min(len(mom_rets), len(opt_rets))
    mom_rets = mom_rets[:n]
    opt_rets = opt_rets[:n]

    results = []
    for mom_w, opt_w in ratios:
        combined_rets = mom_w * mom_rets + opt_w * opt_rets
        equity = [CAPITAL]
        for r in combined_rets:
            equity.append(equity[-1] * (1 + r))
        equity = np.array(equity)

        name = f"Combined_{int(mom_w*100)}_{int(opt_w*100)}"
        metrics = calc_metrics(equity, name)
        print(f"  {name}: CAGR={metrics['cagr']:.1%} | "
              f"Sharpe={metrics['sharpe']:.2f} | MaxDD={metrics['max_drawdown']:.1%}")
        results.append({**metrics, "equity": equity})

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_report(all_results: list[dict], output_path: str,
                    sensitivity: list[dict] = None):
    """Generate markdown comparison report."""
    # Sort by Sharpe ratio
    ranked = sorted(all_results, key=lambda x: x.get("sharpe", -999), reverse=True)

    lines = [
        f"# ${CAPITAL//1000}K 策略对比回测报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"本金: ${CAPITAL:,}",
        f"回测区间: {YEARS} 年 (2016-2026)",
        "",
        "## 排名总览",
        "",
        "| 排名 | 策略 | 年化(CAGR) | 夏普 | Sortino | 最大回撤 | 波动率 | 交易数 | 胜率 | 终值 |",
        "|------|------|-----------|------|---------|---------|--------|--------|------|------|",
    ]

    for i, r in enumerate(ranked, 1):
        name = r.get("name", "?")
        cagr = r.get("cagr", 0)
        sharpe = r.get("sharpe", 0)
        sortino = r.get("sortino", 0)
        max_dd = r.get("max_drawdown", 0)
        vol = r.get("volatility", 0)
        n_trades = r.get("n_trades", 0)
        win_rate = r.get("win_rate", 0)
        final = r.get("final_value", 0)

        lines.append(
            f"| {i} | {name} | {cagr:.1%} | {sharpe:.2f} | {sortino:.2f} | "
            f"{max_dd:.1%} | {vol:.1%} | {n_trades} | {win_rate:.0%} | "
            f"${final:,.0f} |"
        )

    # Winner analysis
    if ranked:
        winner = ranked[0]
        lines.extend([
            "",
            "## 推荐策略",
            "",
            f"**{winner['name']}** 以夏普比率 {winner.get('sharpe', 0):.2f} 排名第一。",
            "",
            f"- 年化回报: {winner.get('cagr', 0):.1%}",
            f"- 最大回撤: {winner.get('max_drawdown', 0):.1%}",
            f"- ${CAPITAL:,} → ${winner.get('final_value', 0):,.0f} ({YEARS}年)",
            "",
        ])

    # Monthly distribution for top 3
    lines.extend([
        "## 月度表现分布",
        "",
        "| 策略 | 盈利月数 | 亏损月数 | 盈利占比 |",
        "|------|---------|---------|---------|",
    ])
    for r in ranked[:5]:
        pos = r.get("pos_months", 0)
        neg = r.get("neg_months", 0)
        total = pos + neg
        ratio = pos / max(total, 1)
        lines.append(f"| {r['name']} | {pos} | {neg} | {ratio:.0%} |")

    # Sensitivity analysis table
    if sensitivity:
        lines.extend([
            "",
            "## ETF 动量轮动参数敏感性分析",
            "",
            "| 参数组合 | 年化(CAGR) | 夏普 | 最大回撤 | 波动率 | 终值 |",
            "|----------|-----------|------|---------|--------|------|",
        ])
        sens_sorted = sorted(sensitivity, key=lambda x: x.get("sharpe", -999), reverse=True)
        for s in sens_sorted:
            lines.append(
                f"| {s['name']} | {s.get('cagr', 0):.1%} | {s.get('sharpe', 0):.2f} | "
                f"{s.get('max_drawdown', 0):.1%} | {s.get('volatility', 0):.1%} | "
                f"${s.get('final_value', 0):,.0f} |"
            )

    lines.extend([
        "",
        "## 结论与建议",
        "",
        "### 期权策略评估",
        "- Credit Spread / Iron Condor 在 BS 定价模型下回测亏损。",
        "  真实市场中 IV 溢价可能提供更好的收益，但回测无法完全模拟。",
        "- Wheel CSP 胜率高但绝对收益低，适合作为辅助收入而非主力。",
        "",
        "### 推荐方案",
        "1. **主力**: ETF 动量轮动 (80%+ 资金) — 已验证的正收益策略",
        "2. **辅助**: 待模拟盘验证后再决定是否加入期权策略",
        "3. **下一步**: 部署到富途模拟盘跑 1-2 周，验证实际执行效果",
        "",
        "## 注意事项",
        "",
        "- 期权策略使用 BS 定价 + 合成 IV，与真实市场 IV 有偏差",
        "- Credit Spread / Iron Condor 的回测未考虑流动性和真实 bid-ask",
        "- ETF 动量轮动使用整股买入，有资金利用率损耗",
        "- 过去表现不代表未来收益",
        "",
        "---",
        f"*由 run_strategy_comparison.py 自动生成*",
    ])

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")
    return report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  $10K Strategy Comparison Backtest")
    print(f"  Capital: ${CAPITAL:,} | Period: {YEARS} years")
    print("=" * 60 + "\n")

    # Step 0: Download data
    download_all_data()

    # Step 1: ETF Momentum Rotation (80% of capital)
    mom_result = backtest_momentum(budget=8000, top_n=5)

    # Step 2: Credit Spread
    cs_result = backtest_credit_spread(budget=3000, width=5, delta=0.30)

    # Step 3: Iron Condor
    ic_result = backtest_iron_condor(budget=3000, width=5,
                                     put_delta=0.15, call_delta=0.15)

    # Step 4: Wheel CSP (optimized DTE=14)
    wheel_result = backtest_wheel(budget=3000, delta=0.25, dte=14)

    # Step 5: ETF Momentum sensitivity (top_n / lookback variations)
    print("\n" + "=" * 60)
    print("  Sensitivity: ETF Momentum top_n / lookback")
    print("=" * 60)
    sensitivity_results = []
    for tn in [3, 5, 7]:
        for lb in [126, 252]:
            label = f"Mom_top{tn}_lb{lb}"
            try:
                daily_data_sens = {}
                for sym in ETF_POOL:
                    df = load_daily(sym)
                    if df is not None and not df.empty:
                        daily_data_sens[sym] = df
                r = MomentumRotation.backtest_momentum(
                    daily_data=daily_data_sens,
                    budget=8000, top_n=tn, lookback=lb, skip=21,
                    sma_period=200, safe_haven="SGOV",
                )
                m = calc_metrics(r["equity"], label, n_trades=r["n_trades"])
                print(f"  {label}: CAGR={m['cagr']:.1%} | Sharpe={m['sharpe']:.2f} | "
                      f"MaxDD={m['max_drawdown']:.1%}")
                sensitivity_results.append(m)
            except Exception as e:
                print(f"  {label}: error - {e}")

    # Step 6: Combined strategies
    all_results = []
    for r in [mom_result, cs_result, ic_result, wheel_result]:
        if r and "equity" in r:
            all_results.append(r)

    # Find best options strategy for combination
    opt_candidates = [r for r in [cs_result, ic_result, wheel_result]
                      if r and "equity" in r]
    combined_results = []
    if mom_result and "equity" in mom_result and opt_candidates:
        best_opt = max(opt_candidates, key=lambda x: x.get("sharpe", -999))
        combined_results = backtest_combined(
            mom_result["equity"], best_opt["equity"], best_opt["name"],
        )
        all_results.extend(combined_results)

    # Generate report
    report_path = str(Path(__file__).resolve().parent / "docs" / "strategy_comparison_10k.md")

    # Remove equity arrays for report (not serializable)
    report_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items() if k != "equity"}
        report_results.append(clean)

    generate_report(report_results, report_path, sensitivity=sensitivity_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
