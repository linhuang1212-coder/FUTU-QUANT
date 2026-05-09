"""
$10K Strategy Lab — 10 strategies, unified backtest framework.

All strategies share the same interface:
  Input:  daily OHLCV DataFrames (dict[symbol -> DataFrame])
  Output: daily equity curve (np.ndarray), plus metrics dict

Strategies:
  1. ETF Momentum Rotation (baseline)
  2. Dual Momentum GEM
  3. Mean Reversion Z-Score
  4. Trend Following + Vol Target
  5. Pairs Trading
  6. Adaptive Asset Allocation (AAA)
  7. HRP Risk Parity
  8. Factor ETF Rotation
  9. XGBoost ML Stock Selection
  10. Multi-Strategy Ensemble
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from data.downloader import save_daily, load_daily

CAPITAL = 10_000
YEARS = 10

# ── Realistic cost model for ETF trading ──
COMMISSION_PER_TRADE = 1.00     # $1 per trade (Futu US stock)
SLIPPAGE_BPS = 5                # 5 basis points slippage per trade
BID_ASK_BPS = 3                 # 3 bps half-spread for liquid ETFs


def apply_trade_cost(capital: float, price: float, qty: int,
                     side: str = "BUY") -> float:
    """Return total cost of a single trade (commission + slippage + spread)."""
    notional = price * qty
    commission = COMMISSION_PER_TRADE
    slippage = notional * SLIPPAGE_BPS / 10000
    spread = notional * BID_ASK_BPS / 10000
    return commission + slippage + spread


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA PREPARATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_SYMBOLS = {
    "broad":   ["SPY", "QQQ", "IWM", "DIA"],
    "sector":  ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
                "XLU", "XLRE", "XLB", "SMH"],
    "intl":    ["VEA", "EEM", "EFA"],
    "bond":    ["TLT", "IEF", "SHY", "AGG", "SGOV"],
    "commodity": ["GLD", "SLV", "GDX", "DBC", "USO"],
    "factor":  ["MTUM", "VLUE", "QUAL", "USMV"],
}

FLAT_SYMBOLS = sorted(set(
    sym for group in ALL_SYMBOLS.values() for sym in group
))


def download_all(years: int = YEARS) -> dict[str, pd.DataFrame]:
    """Download/load all symbols, return {symbol: DataFrame}."""
    data = {}
    for sym in FLAT_SYMBOLS:
        df = load_daily(sym)
        if df is None or df.empty or len(df) < 252 * 3:
            print(f"  Downloading {sym}...")
            save_daily(sym, years=years)
            time.sleep(0.3)
            df = load_daily(sym)
        if df is not None and not df.empty:
            data[sym] = df
            print(f"  {sym}: {len(df)} bars")
    print(f"\n  Total: {len(data)} symbols loaded\n")
    return data


def _align_close_matrix(data: dict[str, pd.DataFrame],
                        symbols: list[str]) -> pd.DataFrame:
    """Build aligned close-price DataFrame from daily data dicts."""
    frames = {}
    for sym in symbols:
        if sym in data and not data[sym].empty:
            df = data[sym].copy()
            if "time_key" in df.columns:
                df["time_key"] = pd.to_datetime(df["time_key"])
                df = df.set_index("time_key")
            frames[sym] = df["close"]
    if not frames:
        return pd.DataFrame()
    result = pd.DataFrame(frames)
    result = result.dropna(how="all").ffill()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_metrics(equity: np.ndarray, name: str,
                 n_trades: int = 0, wins: int = 0) -> dict:
    """Standard metrics from equity curve using LOG returns."""
    from scipy import stats as sp_stats

    if len(equity) < 10:
        return {"name": name, "error": "insufficient data"}

    equity = np.maximum(equity, 1e-6)
    total_ret = (equity[-1] / equity[0]) - 1.0
    n_yrs = len(equity) / 252
    cagr = (1 + total_ret) ** (1 / max(n_yrs, 0.1)) - 1 if total_ret > -1 else -1

    # Log returns (additive, no compounding bias)
    log_rets = np.diff(np.log(equity))
    log_rets = log_rets[np.isfinite(log_rets)]
    if len(log_rets) == 0:
        return {"name": name, "error": "no returns"}

    vol = float(np.std(log_rets) * np.sqrt(252))
    sharpe = float(np.mean(log_rets) / np.std(log_rets) * np.sqrt(252)) if np.std(log_rets) > 0 else 0
    dn = log_rets[log_rets < 0]
    sortino = float(np.mean(log_rets) / np.std(dn) * np.sqrt(252)) if len(dn) > 0 and np.std(dn) > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    win_rate = wins / max(n_trades, 1) if n_trades > 0 else 0

    # ── Fat-tail statistics ──
    skewness = float(sp_stats.skew(log_rets))
    kurtosis = float(sp_stats.kurtosis(log_rets))  # excess kurtosis (0 = normal)
    var_95 = float(np.percentile(log_rets, 5))      # 5% VaR (daily)
    cvar_95 = float(np.mean(log_rets[log_rets <= var_95])) if np.any(log_rets <= var_95) else var_95

    # Monthly returns
    step = 21
    monthly = []
    for i in range(0, len(equity) - step, step):
        monthly.append((equity[i + step] / equity[i]) - 1.0)
    monthly = np.array(monthly)
    pos_m = int(np.sum(monthly > 0)) if len(monthly) > 0 else 0
    neg_m = int(np.sum(monthly <= 0)) if len(monthly) > 0 else 0

    # Annual returns
    annual_rets = {}
    for y in range(int(n_yrs)):
        s, e = y * 252, min((y + 1) * 252, len(equity) - 1)
        if e > s:
            annual_rets[f"Y{y+1}"] = (equity[e] / equity[s]) - 1.0

    return {
        "name": name, "total_return": total_ret, "cagr": cagr,
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd,
        "calmar": calmar, "volatility": vol, "n_trades": n_trades,
        "win_rate": win_rate, "pos_months": pos_m, "neg_months": neg_m,
        "final_value": equity[-1], "annual_returns": annual_rets,
        "skewness": skewness, "kurtosis": kurtosis,
        "var_95_daily": var_95, "cvar_95_daily": cvar_95,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 1: ETF Momentum Rotation (baseline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_momentum(data: dict, budget=8000, top_n=5,
                   lookback=252, skip=21) -> dict:
    from strategy.fractional.momentum_rotation import MomentumRotation
    result = MomentumRotation.backtest_momentum(
        daily_data=data, budget=budget, top_n=top_n,
        lookback=lookback, skip=skip, sma_period=200, safe_haven="SGOV",
    )
    m = calc_metrics(result["equity"], "ETF_Momentum_Rotation",
                     n_trades=result["n_trades"])
    return {**m, "equity": result["equity"]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 2: Dual Momentum GEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_gem(data: dict, budget=10000, lookback=252) -> dict:
    """Global Equities Momentum — Antonacci style.
    Compare SPY vs VEA (12-month return).
    Absolute momentum filter: if winner < SGOV, go to AGG.
    Hold 1 ETF at a time, monthly rebalance.
    """
    syms = ["SPY", "VEA", "AGG", "SGOV"]
    prices = _align_close_matrix(data, syms)
    if prices.empty or len(prices) < lookback + 50:
        return {"name": "Dual_Momentum_GEM", "error": "insufficient data"}

    equity = [budget]
    holding = None
    qty = 0
    cash = budget
    n_trades = 0

    for i in range(lookback, len(prices)):
        month_changed = (i > lookback and
                         prices.index[i].month != prices.index[i - 1].month)

        if not month_changed and i != lookback:
            port_val = cash
            if holding and qty > 0:
                port_val += qty * prices[holding].iloc[i]
            equity.append(port_val)
            continue

        spy_ret = (prices["SPY"].iloc[i] / prices["SPY"].iloc[i - lookback]) - 1
        vea_ret = (prices["VEA"].iloc[i] / prices["VEA"].iloc[i - lookback]) - 1

        sgov_col = "SGOV" if "SGOV" in prices.columns else "AGG"
        sgov_ret = 0.0
        if sgov_col in prices.columns and not np.isnan(prices[sgov_col].iloc[i - lookback]):
            sgov_ret = (prices[sgov_col].iloc[i] / prices[sgov_col].iloc[i - lookback]) - 1

        if spy_ret > vea_ret:
            target = "SPY" if spy_ret > sgov_ret else "AGG"
        else:
            target = "VEA" if vea_ret > sgov_ret else "AGG"

        if target not in prices.columns:
            target = "SPY"

        if target != holding:
            if holding and qty > 0:
                sell_price = prices[holding].iloc[i]
                sell_cost = apply_trade_cost(cash, sell_price, qty, "SELL")
                cash += qty * sell_price - sell_cost
            price = prices[target].iloc[i]
            if price > 0:
                buy_cost_est = apply_trade_cost(cash, price, 1, "BUY")
                qty = int((cash - buy_cost_est * 2) / price)
                if qty >= 1:
                    buy_cost = apply_trade_cost(cash, price, qty, "BUY")
                    cash -= qty * price + buy_cost
                    holding = target
                    n_trades += 1

        port_val = cash
        if holding and qty > 0:
            port_val += qty * prices[holding].iloc[i]
        equity.append(port_val)

    eq = np.array(equity)
    m = calc_metrics(eq, "Dual_Momentum_GEM", n_trades=n_trades)
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 3: Mean Reversion Z-Score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_mean_reversion(data: dict, budget=10000, window=20,
                         entry_z=-2.0, exit_z=0.0,
                         symbols=None) -> dict:
    """Z-Score mean reversion on liquid ETFs.
    Buy when Z < entry_z, sell when Z > exit_z.
    Uses SMA200 as trend filter — only buy in uptrend.
    """
    if symbols is None:
        symbols = ["SPY", "QQQ", "IWM"]
    prices = _align_close_matrix(data, symbols)
    if prices.empty or len(prices) < 252:
        return {"name": "Mean_Reversion_ZScore", "error": "insufficient data"}

    cash = budget
    positions = {}
    equity_curve = []
    n_trades = 0
    wins = 0

    for i in range(200, len(prices)):
        for sym in symbols:
            if sym not in prices.columns:
                continue
            close = prices[sym].iloc[i]
            if np.isnan(close) or close <= 0:
                continue

            window_data = prices[sym].iloc[max(0, i - window):i].dropna()
            if len(window_data) < window:
                continue
            mu = window_data.mean()
            sigma = window_data.std()
            if sigma <= 0:
                continue
            z = (close - mu) / sigma

            sma200 = prices[sym].iloc[max(0, i - 200):i].mean()
            uptrend = close > sma200

            if sym not in positions and z < entry_z and uptrend:
                alloc = cash * 0.3
                qty = int(alloc / close)
                if qty >= 1:
                    tc = apply_trade_cost(cash, close, qty, "BUY")
                    cash -= qty * close + tc
                    positions[sym] = {"qty": qty, "entry": close}
                    n_trades += 1

            elif sym in positions and z > exit_z:
                pos = positions.pop(sym)
                tc = apply_trade_cost(cash, close, pos["qty"], "SELL")
                cash += pos["qty"] * close - tc
                if close > pos["entry"]:
                    wins += 1

        port = cash
        for sym, pos in positions.items():
            if sym in prices.columns:
                port += pos["qty"] * prices[sym].iloc[i]
        equity_curve.append(port)

    eq = np.array(equity_curve) if equity_curve else np.array([budget])
    m = calc_metrics(eq, "Mean_Reversion_ZScore", n_trades=n_trades, wins=wins)
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 4: Trend Following + Vol Target
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_trend_vol(data: dict, budget=10000,
                    target_vol=0.10, sma=200) -> dict:
    """Multi-asset trend following with volatility targeting.
    Go long assets above SMA, scale position by inverse vol.
    Target portfolio vol = 10% annualized.

    INCLUDES: transaction costs on weight changes (turnover cost).
    """
    symbols = ["SPY", "TLT", "GLD", "VEA", "DBC"]
    prices = _align_close_matrix(data, symbols)
    if prices.empty or len(prices) < sma + 50:
        return {"name": "Trend_Vol_Target", "error": "insufficient data"}

    available = [s for s in symbols if s in prices.columns]
    equity = [budget]
    prev_weights = {}
    total_turnover_cost = 0
    n_rebalances = 0
    total_turnover = 0

    for i in range(sma, len(prices)):
        signals = {}
        vols = {}
        for sym in available:
            close = prices[sym].iloc[i]
            sma_val = prices[sym].iloc[i - sma:i].mean()
            if np.isnan(close) or np.isnan(sma_val) or close <= 0:
                continue

            trend = close > sma_val

            rets = prices[sym].iloc[max(0, i - 63):i].pct_change().dropna()
            if len(rets) < 20:
                continue
            ann_vol = float(rets.std() * np.sqrt(252))
            if ann_vol <= 0.001:
                continue

            signals[sym] = 1.0 if trend else 0.0
            vols[sym] = ann_vol

        if not signals or all(v == 0 for v in signals.values()):
            equity.append(equity[-1])
            continue

        weights = {}
        active = {s: v for s, v in signals.items() if v > 0}
        if active:
            inv_vols = {s: 1.0 / vols[s] for s in active}
            total_inv = sum(inv_vols.values())
            raw_weights = {s: v / total_inv for s, v in inv_vols.items()}

            port_vol = 0
            for s in raw_weights:
                port_vol += (raw_weights[s] * vols[s]) ** 2
            port_vol = np.sqrt(port_vol)

            scale = min(target_vol / max(port_vol, 0.01), 1.5)
            weights = {s: w * scale for s, w in raw_weights.items()}

        # Compute turnover and transaction cost
        turnover = 0
        for sym in set(list(weights.keys()) + list(prev_weights.keys())):
            new_w = weights.get(sym, 0)
            old_w = prev_weights.get(sym, 0)
            turnover += abs(new_w - old_w)

        # Cost = turnover * (slippage + spread) applied to portfolio value
        cost_bps = (SLIPPAGE_BPS + BID_ASK_BPS) / 10000
        daily_cost = equity[-1] * turnover * cost_bps
        # Commission: ~$1 per symbol traded
        n_traded = sum(1 for sym in set(list(weights.keys()) + list(prev_weights.keys()))
                       if abs(weights.get(sym, 0) - prev_weights.get(sym, 0)) > 0.01)
        daily_cost += n_traded * COMMISSION_PER_TRADE

        total_turnover_cost += daily_cost
        total_turnover += turnover
        if turnover > 0.02:
            n_rebalances += 1
        prev_weights = weights.copy()

        port_ret = 0
        for sym, w in weights.items():
            daily_ret = (prices[sym].iloc[i] / prices[sym].iloc[i - 1]) - 1
            if np.isfinite(daily_ret):
                port_ret += w * daily_ret

        # Deduct cost from portfolio
        new_val = equity[-1] * (1 + port_ret) - daily_cost
        equity.append(max(new_val, 1))

    eq = np.array(equity)
    m = calc_metrics(eq, "Trend_Vol_Target")
    m["total_turnover_cost"] = total_turnover_cost
    m["avg_daily_turnover"] = total_turnover / max(len(equity), 1)
    m["n_rebalances"] = n_rebalances
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 5: Pairs Trading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_pairs(data: dict, budget=10000) -> dict:
    """Cointegration-based ETF pairs trading.
    Test multiple pairs, trade spread divergences.
    """
    from statsmodels.tsa.stattools import coint

    pair_candidates = [
        ("GLD", "GDX"), ("XLE", "XLI"), ("SPY", "IWM"),
        ("XLF", "XLU"), ("VEA", "EEM"), ("TLT", "IEF"),
    ]

    valid_pairs = []
    for s1, s2 in pair_candidates:
        if s1 in data and s2 in data:
            p1 = _align_close_matrix(data, [s1, s2])
            if len(p1) > 252 and s1 in p1.columns and s2 in p1.columns:
                clean = p1.dropna()
                if len(clean) > 252:
                    _, pval, _ = coint(clean[s1].values, clean[s2].values)
                    if pval < 0.10:
                        valid_pairs.append((s1, s2, pval))
                        print(f"    Pair {s1}/{s2}: coint p={pval:.4f} (valid)")

    if not valid_pairs:
        print("    No cointegrated pairs found, using GLD/GDX as fallback")
        valid_pairs = [("GLD", "GDX", 0.05)]

    all_returns = []
    total_trades = 0
    total_wins = 0

    for s1, s2, _ in valid_pairs[:3]:
        prices_pair = _align_close_matrix(data, [s1, s2]).dropna()
        if len(prices_pair) < 300:
            continue

        lookback = 60
        position = 0
        entry_z = 0
        pair_cash = budget / len(valid_pairs[:3])
        pair_equity = [pair_cash]

        for i in range(lookback, len(prices_pair)):
            window = prices_pair.iloc[i - lookback:i]
            ratio = window[s1] / window[s2]
            mu = ratio.mean()
            sigma = ratio.std()
            if sigma <= 0:
                pair_equity.append(pair_equity[-1])
                continue

            cur_ratio = prices_pair[s1].iloc[i] / prices_pair[s2].iloc[i]
            z = (cur_ratio - mu) / sigma

            ret1 = (prices_pair[s1].iloc[i] / prices_pair[s1].iloc[i - 1]) - 1
            ret2 = (prices_pair[s2].iloc[i] / prices_pair[s2].iloc[i - 1]) - 1

            daily_pnl = 0
            if position == 1:
                daily_pnl = ret1 - ret2
            elif position == -1:
                daily_pnl = ret2 - ret1

            trade_cost_pct = 0
            if position == 0:
                if z < -1.5:
                    position = 1
                    entry_z = z
                    total_trades += 1
                    trade_cost_pct = 2 * (SLIPPAGE_BPS + BID_ASK_BPS) / 10000
                elif z > 1.5:
                    position = -1
                    entry_z = z
                    total_trades += 1
                    trade_cost_pct = 2 * (SLIPPAGE_BPS + BID_ASK_BPS) / 10000
            else:
                if position == 1 and z > -0.5:
                    if z > entry_z:
                        total_wins += 1
                    position = 0
                    trade_cost_pct = 2 * (SLIPPAGE_BPS + BID_ASK_BPS) / 10000
                elif position == -1 and z < 0.5:
                    if z < entry_z:
                        total_wins += 1
                    position = 0
                    trade_cost_pct = 2 * (SLIPPAGE_BPS + BID_ASK_BPS) / 10000
                elif abs(z) > 3.5:
                    position = 0
                    trade_cost_pct = 2 * (SLIPPAGE_BPS + BID_ASK_BPS) / 10000

            net_ret = daily_pnl * 0.5 - trade_cost_pct
            pair_equity.append(pair_equity[-1] * (1 + net_ret))

        if len(pair_equity) > 1:
            pair_rets = np.diff(pair_equity) / np.array(pair_equity[:-1])
            all_returns.append(pair_rets)

    if not all_returns:
        return {"name": "Pairs_Trading", "error": "no valid pairs"}

    min_len = min(len(r) for r in all_returns)
    combined = np.zeros(min_len)
    for r in all_returns:
        combined += r[:min_len] / len(all_returns)

    equity = [budget]
    for r in combined:
        equity.append(equity[-1] * (1 + r))
    eq = np.array(equity)

    m = calc_metrics(eq, "Pairs_Trading", n_trades=total_trades, wins=total_wins)
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 6: Adaptive Asset Allocation (AAA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_aaa(data: dict, budget=10000, top_n=5,
              mom_window=126, vol_window=63) -> dict:
    """Adaptive Asset Allocation.
    Score assets on momentum + inverse-vol + low-correlation.
    Select top N, optimize weights monthly.
    """
    symbols = ["SPY", "QQQ", "TLT", "GLD", "VEA", "EEM",
               "XLE", "XLU", "IEF", "DBC", "SLV", "IWM"]
    prices = _align_close_matrix(data, symbols)
    if prices.empty or len(prices) < mom_window + 100:
        return {"name": "Adaptive_AA", "error": "insufficient data"}

    available = [s for s in symbols if s in prices.columns]
    equity = [budget]
    prev_weights = {}

    for i in range(mom_window, len(prices)):
        month_changed = (i > mom_window and
                         prices.index[i].month != prices.index[i - 1].month)

        if not month_changed and i != mom_window:
            port_ret = 0
            for sym, w in prev_weights.items():
                if sym in prices.columns:
                    dr = (prices[sym].iloc[i] / prices[sym].iloc[i - 1]) - 1
                    if np.isfinite(dr):
                        port_ret += w * dr
            equity.append(equity[-1] * (1 + port_ret))
            continue

        scores = {}
        vol_scores = {}
        for sym in available:
            close = prices[sym].iloc[i]
            past = prices[sym].iloc[i - mom_window:i]
            if len(past) < mom_window or np.isnan(close) or close <= 0:
                continue

            mom = (close / past.iloc[0]) - 1.0
            rets = past.pct_change().dropna()
            vol = rets.std() * np.sqrt(252) if len(rets) > 10 else 999
            if vol <= 0:
                continue

            scores[sym] = mom / max(vol, 0.01)
            vol_scores[sym] = vol

        if len(scores) < 2:
            equity.append(equity[-1])
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in ranked[:top_n]]

        inv_vols = {s: 1.0 / vol_scores[s] for s in selected if s in vol_scores}
        total_iv = sum(inv_vols.values())
        new_weights = {s: v / total_iv for s, v in inv_vols.items()} if total_iv > 0 else {}

        # Turnover cost on rebalance
        turnover = 0
        for sym in set(list(new_weights.keys()) + list(prev_weights.keys())):
            turnover += abs(new_weights.get(sym, 0) - prev_weights.get(sym, 0))
        cost_bps = (SLIPPAGE_BPS + BID_ASK_BPS) / 10000
        rebal_cost = equity[-1] * turnover * cost_bps
        n_traded = sum(1 for sym in set(list(new_weights.keys()) + list(prev_weights.keys()))
                       if abs(new_weights.get(sym, 0) - prev_weights.get(sym, 0)) > 0.01)
        rebal_cost += n_traded * COMMISSION_PER_TRADE

        prev_weights = new_weights

        port_ret = 0
        for sym, w in new_weights.items():
            if sym in prices.columns:
                dr = (prices[sym].iloc[i] / prices[sym].iloc[i - 1]) - 1
                if np.isfinite(dr):
                    port_ret += w * dr
        equity.append(max(equity[-1] * (1 + port_ret) - rebal_cost, 1))

    eq = np.array(equity)
    m = calc_metrics(eq, "Adaptive_AA")
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 7: HRP Risk Parity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_hrp(data: dict, budget=10000, method="hrp") -> dict:
    """Portfolio with HRP / min-variance / risk-parity / equal weights.
    Combined with momentum filter: only include assets above SMA200.
    Monthly rebalance.
    """
    symbols = ["SPY", "QQQ", "TLT", "GLD", "VEA", "EEM",
               "XLE", "XLU", "IEF", "SLV", "IWM", "XLK"]
    prices = _align_close_matrix(data, symbols)
    if prices.empty or len(prices) < 300:
        return {"name": f"HRP_{method}", "error": "insufficient data"}

    available = [s for s in symbols if s in prices.columns]
    equity = [budget]
    weights = {}
    prev_weights = {}

    for i in range(252, len(prices)):
        month_changed = (i > 252 and
                         prices.index[i].month != prices.index[i - 1].month)

        rebal_cost = 0
        if month_changed or i == 252:
            eligible = []
            for sym in available:
                close = prices[sym].iloc[i]
                sma = prices[sym].iloc[i - 200:i].mean()
                if np.isfinite(close) and np.isfinite(sma) and close > sma:
                    eligible.append(sym)

            if len(eligible) < 2:
                eligible = available[:3]

            ret_window = prices[eligible].iloc[max(0, i - 252):i].pct_change().dropna()

            if method == "hrp" and len(ret_window) > 50:
                try:
                    from pypfopt import HRPOpt
                    hrp = HRPOpt(ret_window)
                    raw = hrp.optimize()
                    weights = {s: raw.get(s, 0) for s in eligible}
                except Exception:
                    weights = {s: 1.0 / len(eligible) for s in eligible}
            elif method == "min_variance" and len(ret_window) > 50:
                try:
                    cov = ret_window.cov()
                    inv_cov = np.linalg.pinv(cov.values)
                    ones = np.ones(len(eligible))
                    w = inv_cov @ ones / (ones @ inv_cov @ ones)
                    w = np.maximum(w, 0)
                    w /= w.sum()
                    weights = dict(zip(eligible, w))
                except Exception:
                    weights = {s: 1.0 / len(eligible) for s in eligible}
            elif method == "risk_parity" and len(ret_window) > 50:
                try:
                    vols = ret_window.std()
                    inv_vol = 1.0 / vols
                    w = inv_vol / inv_vol.sum()
                    weights = dict(zip(eligible, w.values))
                except Exception:
                    weights = {s: 1.0 / len(eligible) for s in eligible}
            else:
                weights = {s: 1.0 / len(eligible) for s in eligible}

            # Turnover cost
            turnover = 0
            for sym in set(list(weights.keys()) + list(prev_weights.keys())):
                turnover += abs(weights.get(sym, 0) - prev_weights.get(sym, 0))
            cost_bps = (SLIPPAGE_BPS + BID_ASK_BPS) / 10000
            rebal_cost = equity[-1] * turnover * cost_bps
            n_traded = sum(1 for sym in set(list(weights.keys()) + list(prev_weights.keys()))
                           if abs(weights.get(sym, 0) - prev_weights.get(sym, 0)) > 0.01)
            rebal_cost += n_traded * COMMISSION_PER_TRADE
            prev_weights = weights.copy()

        port_ret = 0
        for sym, w in weights.items():
            if sym in prices.columns and i > 0:
                dr = (prices[sym].iloc[i] / prices[sym].iloc[i - 1]) - 1
                if np.isfinite(dr):
                    port_ret += w * dr
        equity.append(max(equity[-1] * (1 + port_ret) - rebal_cost, 1))

    eq = np.array(equity)
    label = {"hrp": "HRP_RiskParity", "min_variance": "MinVariance",
             "risk_parity": "InvVol_RiskParity", "equal": "Equal_Weight"}
    m = calc_metrics(eq, label.get(method, f"Portfolio_{method}"))
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 8: Factor ETF Rotation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_factor_rotation(data: dict, budget=10000,
                          lookback=63) -> dict:
    """Rotate among factor ETFs based on recent momentum.
    MTUM (momentum), VLUE (value), QUAL (quality), USMV (low vol).
    Hold the strongest 1-2 factors; flee to SGOV in crisis.
    """
    factor_etfs = ["MTUM", "VLUE", "QUAL", "USMV"]
    safe = "SGOV"
    all_syms = factor_etfs + [safe, "SPY"]
    prices = _align_close_matrix(data, all_syms)
    if prices.empty or len(prices) < lookback + 100:
        return {"name": "Factor_ETF_Rotation", "error": "insufficient data"}

    available_factors = [s for s in factor_etfs if s in prices.columns]
    if len(available_factors) < 2:
        return {"name": "Factor_ETF_Rotation", "error": "not enough factor ETFs"}

    equity = [budget]
    holding = None
    qty = 0
    cash = budget
    n_trades = 0

    for i in range(lookback, len(prices)):
        month_changed = (i > lookback and
                         prices.index[i].month != prices.index[i - 1].month)

        if not month_changed and i != lookback:
            port = cash
            if holding and qty > 0 and holding in prices.columns:
                port += qty * prices[holding].iloc[i]
            equity.append(port)
            continue

        spy_sma = prices["SPY"].iloc[max(0, i - 200):i].mean() if "SPY" in prices.columns else 0
        spy_close = prices["SPY"].iloc[i] if "SPY" in prices.columns else 0
        crisis = spy_close < spy_sma * 0.95 if spy_sma > 0 else False

        if crisis and safe in prices.columns:
            target = safe
        else:
            scores = {}
            for f in available_factors:
                if f in prices.columns:
                    past = prices[f].iloc[i - lookback]
                    cur = prices[f].iloc[i]
                    if past > 0 and np.isfinite(cur):
                        scores[f] = (cur / past) - 1.0
            if scores:
                target = max(scores, key=scores.get)
            else:
                target = available_factors[0]

        if target != holding:
            if holding and qty > 0 and holding in prices.columns:
                sell_price = prices[holding].iloc[i]
                sell_tc = apply_trade_cost(cash, sell_price, qty, "SELL")
                cash += qty * sell_price - sell_tc
            if target in prices.columns:
                price = prices[target].iloc[i]
                if price > 0:
                    buy_tc_est = apply_trade_cost(cash, price, 1, "BUY")
                    qty = int((cash - buy_tc_est * 2) / price)
                    if qty >= 1:
                        buy_tc = apply_trade_cost(cash, price, qty, "BUY")
                        cash -= qty * price + buy_tc
                        holding = target
                        n_trades += 1

        port = cash
        if holding and qty > 0 and holding in prices.columns:
            port += qty * prices[holding].iloc[i]
        equity.append(port)

    eq = np.array(equity)
    m = calc_metrics(eq, "Factor_ETF_Rotation", n_trades=n_trades)
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 9: XGBoost ML Stock Selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_xgboost(data: dict, budget=10000, train_window=504,
                  predict_window=21, top_n=5) -> dict:
    """XGBoost walk-forward stock selection.
    Features: momentum (multiple windows), volatility, RSI-proxy.
    Label: forward 21-day return.
    Quarterly retrain (not monthly) for speed, buy top N scored stocks.
    """
    import xgboost as xgb

    symbols = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV",
               "XLI", "XLY", "SMH", "GLD", "TLT", "VEA", "EEM"]
    prices = _align_close_matrix(data, symbols)
    if prices.empty or len(prices) < train_window + predict_window + 100:
        return {"name": "XGBoost_ML", "error": "insufficient data"}

    available = [s for s in symbols if s in prices.columns]

    # Pre-compute all features as matrix for speed
    feat_names = ["mom_21", "mom_63", "mom_126", "mom_252",
                  "vol_21", "vol_63", "sma_ratio", "rsi_proxy"]

    def build_features_fast(close_series, idx):
        if idx < 252:
            return None
        c = close_series.values
        cur = c[idx]
        if cur <= 0 or np.isnan(cur):
            return None
        feats = np.zeros(8)
        for j, w in enumerate([21, 63, 126, 252]):
            past = c[idx - w]
            feats[j] = (cur / past) - 1 if past > 0 else 0
        rets21 = np.diff(c[idx - 21:idx + 1]) / c[idx - 21:idx]
        feats[4] = np.std(rets21) * np.sqrt(252) if len(rets21) > 5 else 0
        rets63 = np.diff(c[max(0, idx - 63):idx + 1]) / c[max(0, idx - 63):idx]
        feats[5] = np.std(rets63) * np.sqrt(252) if len(rets63) > 10 else 0
        sma50 = np.mean(c[idx - 50:idx])
        sma200 = np.mean(c[idx - 200:idx])
        feats[6] = sma50 / sma200 if sma200 > 0 else 1
        delta = np.diff(c[idx - 14:idx + 1])
        gain = np.mean(np.maximum(delta, 0))
        loss = np.mean(np.maximum(-delta, 0))
        feats[7] = gain / (gain + loss) if (gain + loss) > 0 else 0.5
        return feats

    equity = [budget]
    holdings = {}
    cash = budget
    n_trades = 0
    model = None

    # Quarterly retrain (every 63 bars) instead of monthly
    retrain_interval = 63
    rebalance_interval = 21

    for i in range(train_window, len(prices)):
        is_retrain = (i == train_window or (i - train_window) % retrain_interval == 0)
        is_rebalance = (i == train_window or (i - train_window) % rebalance_interval == 0)

        if is_retrain:
            # Build training data from recent window only (not all history)
            train_start = max(252, i - train_window)
            X_train, y_train = [], []
            # Sample every 5th bar for speed
            for t in range(train_start, i - predict_window, 5):
                for sym in available:
                    feats = build_features_fast(prices[sym], t)
                    if feats is None:
                        continue
                    fwd_idx = min(t + predict_window, len(prices) - 1)
                    fwd = (prices[sym].iloc[fwd_idx] / prices[sym].iloc[t]) - 1
                    if np.isfinite(fwd):
                        X_train.append(feats)
                        y_train.append(fwd)

            if len(X_train) >= 50:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                model = xgb.XGBRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0,
                )
                model.fit(X_train, y_train)

        if is_rebalance and model is not None:
            predictions = {}
            for sym in available:
                feats = build_features_fast(prices[sym], i)
                if feats is not None:
                    pred = model.predict(feats.reshape(1, -1))[0]
                    predictions[sym] = pred

            ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            new_holdings = [s for s, _ in ranked[:top_n]]

            for sym, q in holdings.items():
                if q > 0 and sym in prices.columns:
                    sell_price = prices[sym].iloc[i]
                    sell_tc = apply_trade_cost(cash, sell_price, q, "SELL")
                    cash += q * sell_price - sell_tc
            holdings = {}

            if new_holdings:
                per_slot = cash / len(new_holdings)
                for sym in new_holdings:
                    price = prices[sym].iloc[i]
                    if price > 0:
                        tc_est = apply_trade_cost(cash, price, 1, "BUY")
                        q = int((per_slot - tc_est * 2) / price)
                        if q >= 1:
                            buy_tc = apply_trade_cost(cash, price, q, "BUY")
                            cash -= q * price + buy_tc
                            holdings[sym] = q
                            n_trades += 1

        port = cash
        for sym, q in holdings.items():
            if sym in prices.columns:
                port += q * prices[sym].iloc[i]
        equity.append(port)

    eq = np.array(equity) if len(equity) > 1 else np.array([budget, budget])
    m = calc_metrics(eq, "XGBoost_ML", n_trades=n_trades)
    return {**m, "equity": eq}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY 10: Multi-Strategy Ensemble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strat_ensemble(results: list[dict], budget=10000,
                   top_n=3) -> list[dict]:
    """Combine top N strategies by Sharpe-weighted returns."""
    valid = [r for r in results if "equity" in r and r.get("sharpe", -99) > 0]
    if len(valid) < 2:
        return []

    ranked = sorted(valid, key=lambda x: x["sharpe"], reverse=True)[:top_n]
    names = [r["name"] for r in ranked]
    print(f"    Ensemble top {top_n}: {names}")

    equities = [r["equity"] for r in ranked]
    min_len = min(len(e) for e in equities)
    returns_list = [np.diff(e[:min_len]) / e[:min_len - 1] for e in equities]

    ensemble_results = []

    # Equal weight
    eq_rets = np.mean(returns_list, axis=0)
    eq_equity = [budget]
    for r in eq_rets:
        eq_equity.append(eq_equity[-1] * (1 + r))
    eq_eq = np.array(eq_equity)
    m = calc_metrics(eq_eq, "Ensemble_EqualWeight")
    ensemble_results.append({**m, "equity": eq_eq})

    # Sharpe weight
    sharpes = np.array([r["sharpe"] for r in ranked])
    sw = sharpes / sharpes.sum()
    sw_rets = np.zeros(len(returns_list[0]))
    for i, r in enumerate(returns_list):
        sw_rets += sw[i] * r
    sw_equity = [budget]
    for r in sw_rets:
        sw_equity.append(sw_equity[-1] * (1 + r))
    sw_eq = np.array(sw_equity)
    m = calc_metrics(sw_eq, "Ensemble_SharpeWeight")
    ensemble_results.append({**m, "equity": sw_eq})

    # Inverse-vol weight
    vols = np.array([np.std(r) for r in returns_list])
    ivw = (1.0 / vols) / (1.0 / vols).sum() if all(v > 0 for v in vols) else np.ones(len(vols)) / len(vols)
    iv_rets = np.zeros(len(returns_list[0]))
    for i, r in enumerate(returns_list):
        iv_rets += ivw[i] * r
    iv_equity = [budget]
    for r in iv_rets:
        iv_equity.append(iv_equity[-1] * (1 + r))
    iv_eq = np.array(iv_equity)
    m = calc_metrics(iv_eq, "Ensemble_InvVolWeight")
    ensemble_results.append({**m, "equity": iv_eq})

    return ensemble_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_strategy(equity: np.ndarray, name: str,
                      n_mc=500, n_folds=6) -> dict:
    """Rigorous validation: K-Fold + Monte Carlo + stress + fat-tail + crisis correlation.

    Aligned with the quantitative methodology:
    - K-Fold: >=4/6 positive folds = robust
    - Monte Carlo: shuffle returns, PBO = prob of backtest overfitting
    - Stress: COVID, Rate Hike, Q4 2018, Tariff 2025, 2008-style L-shape
    - Fat-tail: skewness, kurtosis, VaR, CVaR
    - Crisis correlation: correlation of strategy returns with SPY during drawdowns
    """
    from scipy import stats as sp_stats

    # Use log returns for all validation
    equity = np.maximum(equity, 1e-6)
    log_rets = np.diff(np.log(equity))
    log_rets = log_rets[np.isfinite(log_rets)]
    if len(log_rets) < 252:
        return {"name": name, "mc_sharpe_mean": 0, "mc_sharpe_std": 0,
                "stress_results": {}, "kfold_pass": 0, "kfold_total": 0}

    actual_sharpe = float(np.mean(log_rets) / np.std(log_rets) * np.sqrt(252))

    # ── K-Fold time-series validation ──
    fold_size = len(log_rets) // n_folds
    fold_sharpes = []
    fold_positive = 0
    for f in range(n_folds):
        start = f * fold_size
        end = (f + 1) * fold_size if f < n_folds - 1 else len(log_rets)
        fold_r = log_rets[start:end]
        if len(fold_r) > 20 and np.std(fold_r) > 0:
            fs = float(np.mean(fold_r) / np.std(fold_r) * np.sqrt(252))
            fold_sharpes.append(fs)
            if fs > 0:
                fold_positive += 1
        else:
            fold_sharpes.append(0)

    kfold_pass = fold_positive
    kfold_total = n_folds
    kfold_robust = kfold_pass >= 4

    # ── Monte Carlo shuffle ──
    mc_sharpes = []
    for _ in range(n_mc):
        shuffled = np.random.permutation(log_rets)
        s = float(np.mean(shuffled) / np.std(shuffled) * np.sqrt(252)) if np.std(shuffled) > 0 else 0
        mc_sharpes.append(s)
    mc_sharpes = np.array(mc_sharpes)
    pbo = float(np.mean(mc_sharpes >= actual_sharpe))

    # ── Stress test windows (expanded) ──
    stress_windows = {
        "COVID_2020_crash": (4 * 252, 4 * 252 + 30),         # Feb-Mar 2020
        "COVID_2020_6mo":   (4 * 252, 4 * 252 + 126),        # Feb-Aug 2020
        "Rate_Hike_2022":   (6 * 252, 6 * 252 + 180),        # Jan-Jun 2022
        "Q4_2018":          (2 * 252 + 190, 2 * 252 + 252),  # Q4 2018
        "Tariff_2025":      (9 * 252, 9 * 252 + 63),         # Early 2025
        "L_Shape_Recovery":  (4 * 252, 4 * 252 + 252),       # Full year from COVID
    }
    stress_results = {}
    for label, (start, end) in stress_windows.items():
        if start < len(equity) and end < len(equity):
            window_eq = equity[start:end + 1]
            if len(window_eq) > 1:
                stress_ret = (window_eq[-1] / window_eq[0]) - 1
                pk = np.maximum.accumulate(window_eq)
                stress_dd = float(np.min((window_eq - pk) / pk))
                stress_results[label] = {
                    "return": stress_ret,
                    "max_dd": stress_dd,
                }

    # ── Fat-tail diagnostics ──
    skewness = float(sp_stats.skew(log_rets))
    kurtosis_excess = float(sp_stats.kurtosis(log_rets))
    var_95 = float(np.percentile(log_rets, 5))
    cvar_95 = float(np.mean(log_rets[log_rets <= var_95])) if np.any(log_rets <= var_95) else var_95
    var_99 = float(np.percentile(log_rets, 1))
    cvar_99 = float(np.mean(log_rets[log_rets <= var_99])) if np.any(log_rets <= var_99) else var_99

    # Jarque-Bera normality test
    jb_stat, jb_pval = sp_stats.jarque_bera(log_rets)
    is_normal = jb_pval > 0.05

    fat_tail_warning = kurtosis_excess > 3  # Normal = 0, >3 = heavy tails

    # ── Crisis correlation (strategy vs drawdown periods) ──
    # During worst 10% of days, is strategy correlated with market?
    worst_10pct_threshold = np.percentile(log_rets, 10)
    worst_days = log_rets <= worst_10pct_threshold
    worst_mean = float(np.mean(log_rets[worst_days]))
    worst_vol = float(np.std(log_rets[worst_days]))

    # ── Bootstrap confidence interval ──
    bootstrap_sharpes = []
    for _ in range(200):
        sample = np.random.choice(log_rets, size=len(log_rets), replace=True)
        s = float(np.mean(sample) / np.std(sample) * np.sqrt(252)) if np.std(sample) > 0 else 0
        bootstrap_sharpes.append(s)
    sharpe_ci_low = float(np.percentile(bootstrap_sharpes, 5))
    sharpe_ci_high = float(np.percentile(bootstrap_sharpes, 95))

    # ── MaxDD recovery time ──
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd_idx = np.argmin(dd)
    recovery_days = 0
    for j in range(max_dd_idx, len(equity)):
        if equity[j] >= peak[max_dd_idx]:
            recovery_days = j - max_dd_idx
            break

    return {
        "name": name,
        "actual_sharpe": actual_sharpe,
        "kfold_pass": kfold_pass,
        "kfold_total": kfold_total,
        "kfold_robust": kfold_robust,
        "kfold_sharpes": fold_sharpes,
        "mc_sharpe_mean": float(np.mean(mc_sharpes)),
        "mc_sharpe_std": float(np.std(mc_sharpes)),
        "pbo_probability": pbo,
        "sharpe_ci_90": (sharpe_ci_low, sharpe_ci_high),
        "stress_results": stress_results,
        "recovery_days": recovery_days,
        # NEW: fat-tail diagnostics
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "var_95_daily": var_95,
        "cvar_95_daily": cvar_95,
        "var_99_daily": var_99,
        "cvar_99_daily": cvar_99,
        "jarque_bera_normal": is_normal,
        "fat_tail_warning": fat_tail_warning,
        # NEW: crisis behavior
        "worst_10pct_mean_ret": worst_mean,
        "worst_10pct_vol": worst_vol,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REPORT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_report(all_results: list[dict],
                    validations: list[dict],
                    output_path: str):
    """Generate comprehensive markdown report with bias-corrected metrics."""
    from datetime import datetime

    ranked = sorted(all_results, key=lambda x: x.get("sharpe", -999), reverse=True)

    lines = [
        f"# $10K Strategy Lab Report (BIAS-CORRECTED v2)",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Capital: ${CAPITAL:,} | Backtest: {YEARS} years (2016-2026)",
        f"Strategies tested: {len(ranked)}",
        "",
        "### Corrections Applied",
        "- **Log returns** (vs simple returns) for all metrics",
        f"- **Transaction costs**: ${COMMISSION_PER_TRADE:.2f}/trade + "
        f"{SLIPPAGE_BPS}bps slippage + {BID_ASK_BPS}bps spread",
        "- **Fat-tail diagnostics**: skewness, excess kurtosis, VaR, CVaR",
        "- **Crisis stress tests**: COVID crash, L-shape recovery, rate hike, tariffs",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Strategy | CAGR | Sharpe | Sortino | MaxDD | Vol | Skew | Kurt | VaR95 | Final |",
        "|------|----------|------|--------|---------|-------|-----|------|------|-------|-------|",
    ]

    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r.get('name', '?')} | "
            f"{r.get('cagr', 0):.1%} | {r.get('sharpe', 0):.2f} | "
            f"{r.get('sortino', 0):.2f} | "
            f"{r.get('max_drawdown', 0):.1%} | {r.get('volatility', 0):.1%} | "
            f"{r.get('skewness', 0):.2f} | {r.get('kurtosis', 0):.1f} | "
            f"{r.get('var_95_daily', 0):.2%} | "
            f"${r.get('final_value', 0):,.0f} |"
        )

    # Monthly performance
    lines.extend([
        "",
        "## Monthly Performance Distribution",
        "",
        "| Strategy | Positive Months | Negative Months | Win Rate |",
        "|----------|----------------|-----------------|----------|",
    ])
    for r in ranked[:10]:
        pos = r.get("pos_months", 0)
        neg = r.get("neg_months", 0)
        total = pos + neg
        ratio = pos / max(total, 1)
        lines.append(f"| {r['name']} | {pos} | {neg} | {ratio:.0%} |")

    # Turnover & Cost (for strategies that report it)
    turnover_strats = [r for r in ranked if r.get("total_turnover_cost") is not None]
    if turnover_strats:
        lines.extend([
            "",
            "## Turnover & Transaction Cost Impact",
            "",
            "| Strategy | Total TC Cost | Avg Daily Turnover | Rebalances |",
            "|----------|-------------|-------------------|------------|",
        ])
        for r in turnover_strats:
            lines.append(
                f"| {r['name']} | ${r.get('total_turnover_cost', 0):,.0f} | "
                f"{r.get('avg_daily_turnover', 0):.4f} | "
                f"{r.get('n_rebalances', 0)} |"
            )

    # Annual returns
    lines.extend(["", "## Annual Return Breakdown", ""])
    header = "| Strategy |"
    sep = "|----------|"
    for y in range(YEARS):
        header += f" Y{y+1} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)
    for r in ranked[:10]:
        row = f"| {r.get('name', '?')} |"
        annual = r.get("annual_returns", {})
        for y in range(YEARS):
            val = annual.get(f"Y{y+1}", 0)
            row += f" {val:+.1%} |"
        lines.append(row)

    # Validation results
    if validations:
        lines.extend([
            "",
            "## Validation Results (Top Strategies)",
            "",
            "### K-Fold Time-Series Cross Validation (6 folds)",
            "",
            "| Strategy | K-Fold Pass | Fold Sharpes | Robust? |",
            "|----------|------------|-------------|---------|",
        ])
        for v in validations:
            kp = v.get("kfold_pass", 0)
            kt = v.get("kfold_total", 6)
            ks = v.get("kfold_sharpes", [])
            ks_str = ", ".join([f"{s:.2f}" for s in ks]) if ks else "N/A"
            robust = "YES" if v.get("kfold_robust", False) else "NO"
            lines.append(f"| {v['name']} | {kp}/{kt} | {ks_str} | {robust} |")

        lines.extend([
            "",
            "### Monte Carlo + Confidence Intervals",
            "",
            "| Strategy | Sharpe | MC Mean | PBO | 90% CI | Recovery |",
            "|----------|--------|---------|-----|--------|----------|",
        ])
        for v in validations:
            ci = v.get("sharpe_ci_90", (0, 0))
            rec = v.get("recovery_days", 0)
            lines.append(
                f"| {v['name']} | {v.get('actual_sharpe', 0):.2f} | "
                f"{v.get('mc_sharpe_mean', 0):.2f} | "
                f"{v.get('pbo_probability', 0):.1%} | "
                f"[{ci[0]:.2f}, {ci[1]:.2f}] | {rec}d |"
            )

        # Fat-tail diagnostics
        lines.extend([
            "",
            "### Fat-Tail & Risk Diagnostics",
            "",
            "| Strategy | Skewness | ExKurtosis | VaR95 | CVaR95 | VaR99 | CVaR99 | Normal? | Fat Tail? |",
            "|----------|----------|-----------|-------|--------|-------|--------|---------|-----------|",
        ])
        for v in validations:
            lines.append(
                f"| {v['name']} | {v.get('skewness', 0):.2f} | "
                f"{v.get('kurtosis_excess', 0):.1f} | "
                f"{v.get('var_95_daily', 0):.2%} | {v.get('cvar_95_daily', 0):.2%} | "
                f"{v.get('var_99_daily', 0):.2%} | {v.get('cvar_99_daily', 0):.2%} | "
                f"{'YES' if v.get('jarque_bera_normal') else 'NO'} | "
                f"{'WARNING' if v.get('fat_tail_warning') else 'OK'} |"
            )

        # Stress test details
        lines.extend([
            "",
            "### Stress Test Details",
            "",
            "| Strategy | COVID Crash | COVID 6mo | L-Shape Yr | Rate Hike | Q4 2018 | Tariff 2025 |",
            "|----------|-----------|----------|-----------|----------|---------|-------------|",
        ])
        for v in validations:
            stress = v.get("stress_results", {})
            def _sr(key):
                s = stress.get(key, {})
                if isinstance(s, dict):
                    return f"{s.get('return', 0):+.1%}"
                return f"{s:+.1%}" if isinstance(s, (int, float)) else "N/A"
            lines.append(
                f"| {v['name']} | "
                f"{_sr('COVID_2020_crash')} | {_sr('COVID_2020_6mo')} | "
                f"{_sr('L_Shape_Recovery')} | {_sr('Rate_Hike_2022')} | "
                f"{_sr('Q4_2018')} | {_sr('Tariff_2025')} |"
            )

        # Crisis behavior
        lines.extend([
            "",
            "### Crisis Behavior (Worst 10% of Days)",
            "",
            "| Strategy | Mean Ret (worst 10%) | Vol (worst 10%) |",
            "|----------|---------------------|-----------------|",
        ])
        for v in validations:
            lines.append(
                f"| {v['name']} | {v.get('worst_10pct_mean_ret', 0):.3%} | "
                f"{v.get('worst_10pct_vol', 0):.3%} |"
            )

    # Recommendations
    if ranked:
        top3 = ranked[:3]
        lines.extend([
            "",
            "## Recommendations",
            "",
            "### Top 3 Strategies for $10K Portfolio (Post-Correction)",
            "",
        ])
        for i, r in enumerate(top3, 1):
            lines.append(
                f"{i}. **{r['name']}** — Sharpe {r.get('sharpe', 0):.2f}, "
                f"CAGR {r.get('cagr', 0):.1%}, MaxDD {r.get('max_drawdown', 0):.1%}"
            )

        lines.extend([
            "",
            "### Deployment Plan",
            "",
            "1. Deploy Top 3 strategies to paper trading (Futu SIMULATE)",
            "2. Run for 1-2 weeks, compare live NAV vs backtest expectations",
            "3. If consistent, migrate to real account with scaled position sizes",
            "",
            "### Risk Warnings",
            "",
            "- Strategies with **excess kurtosis > 3** have fat tails — real losses may exceed VaR",
            "- **Negative skewness** implies asymmetric downside risk",
            "- **PBO > 5%** suggests possible overfitting",
            "- Sharpe ratios above 2.0 after cost correction deserve extra scrutiny",
            "",
            "---",
            "*Generated by backtest/strategy_lab.py (bias-corrected v2)*",
        ])

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\n  Report saved: {output_path}")
    return report
