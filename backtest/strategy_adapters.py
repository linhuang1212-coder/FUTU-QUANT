from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  Validation Task definition
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ValidationTask:
    """Unified task definition for 7-Gate validation."""
    name: str
    strategy_type: str  # "options" | "equity" | "etf"
    symbols: list[str] = field(default_factory=list)

    backtest_func: Optional[Callable] = None
    pnl_for_cpcv: Optional[Callable] = None
    pnl_for_noise: Optional[Callable] = None
    eval_for_sensitivity: Optional[Callable] = None
    stress_func: Optional[Callable] = None

    default_params: dict = field(default_factory=dict)
    param_grid: dict = field(default_factory=dict)

    economic_thesis: str = ""
    needs_ivr: bool = False
    needs_intraday: bool = False
    capital: float = 3000.0


def _sharpe(pnls):
    if len(pnls) < 3:
        return 0.0
    arr = np.array(pnls, dtype=float)
    s = np.std(arr)
    if s <= 0:
        return 0.0
    return float(np.mean(arr) / s * np.sqrt(12))


# ═══════════════════════════════════════════════════════════════════
#  Economic theses
# ═══════════════════════════════════════════════════════════════════

ECONOMIC_THESES = {
    "credit_spread": (
        "Volatility Risk Premium: options are systematically overpriced relative to "
        "realized volatility. Selling OTM put spreads when IVR is elevated captures "
        "the IV-RV spread. Academic evidence: Coval & Shumway (2001), Bakshi & "
        "Kapadia (2003). Edge: systematic theta collection + mean reversion of IV."
    ),
    "straddle_squeeze": (
        "Volatility mean reversion: periods of abnormally low realized volatility "
        "tend to be followed by volatility expansion. Buying straddles during BB "
        "squeeze captures the subsequent move. Academic evidence: volatility "
        "clustering (Mandelbrot 1963, Engle 1982 ARCH). Edge: buying vol cheaply "
        "when markets are complacent."
    ),
    "earnings_spread": (
        "Event-driven volatility: earnings announcements create information "
        "asymmetry and demand for options. IV expansion before events often exceeds "
        "realized moves, but buying early enough captures the IV run-up. Edge: "
        "systematic event premium harvesting with defined risk."
    ),
    "wheel_csp": (
        "Cash-secured put selling: collect premium by selling puts on stocks you'd "
        "be willing to own. Combines volatility risk premium harvesting with value "
        "entry. Academic evidence: Coval & Shumway (2001). Edge: systematic income "
        "generation with stock ownership as downside."
    ),
    "momentum_rotation": (
        "Cross-sectional momentum: assets with strong recent returns continue "
        "to outperform. 12M-1M momentum avoids short-term reversal noise. "
        "SMA200 trend filter reduces drawdowns during bear markets. Academic "
        "evidence: Jegadeesh & Titman (1993), Moskowitz et al. (2012). "
        "Edge: systematic risk premia harvesting across asset classes."
    ),
    "momentum": (
        "Time-series and cross-sectional momentum: winners continue winning. "
        "Jegadeesh & Titman (1993). Edge: systematic trend following with risk "
        "management."
    ),
    "mean_reversion": (
        "Short-term mean reversion: overreaction to news causes temporary price "
        "displacement. Poterba & Summers (1988). Edge: statistical tendency for "
        "prices to revert to moving averages."
    ),
    "breakout": (
        "Trend continuation after range breakout: consolidation patterns resolve "
        "with momentum. Edge: capturing the transition from low to high volatility "
        "regimes."
    ),
    "rsi_reversal": (
        "RSI oversold/overbought reversal: extreme RSI readings signal exhaustion. "
        "Wilder (1978). Edge: contrarian entry at statistical extremes."
    ),
    "multi_factor": (
        "Factor combination: multiple weak signals aggregated produce robust "
        "composite signal. Edge: diversification across alpha sources reduces "
        "single-factor risk."
    ),
}


# ═══════════════════════════════════════════════════════════════════
#  Lazy imports — wrapped so missing modules don't crash the file
# ═══════════════════════════════════════════════════════════════════

def _import_turbo_core():
    from backtest.turbo_core import sim_spread_batch, precompute_ivr_fast
    return sim_spread_batch, precompute_ivr_fast


def _import_full_validation():
    from backtest.full_validation import sim_spread_with_cost, sim_wheel_with_cost, CostModel
    return sim_spread_with_cost, sim_wheel_with_cost, CostModel


def _import_bs():
    from options.pricer import bs_price
    return bs_price


def _import_bb_width():
    from options.pricer import bb_width
    return bb_width


def _import_synth_iv():
    from options.backtest import _synth_iv, _hist_vol
    return _synth_iv, _hist_vol


# ═══════════════════════════════════════════════════════════════════
#  Builder: Credit Spread
# ═══════════════════════════════════════════════════════════════════

def _cs_backtest(closes, spread_width=5, target_delta=0.25, max_hold=21,
                 tp_pct=0.50, sl_pct=2.0, min_ivr=30, r=0.05):
    """Run credit spread backtest, returning list of PnL floats."""
    try:
        sim_spread_batch, precompute_ivr_fast = _import_turbo_core()
        ivr_arr = precompute_ivr_fast(closes)
        results = sim_spread_batch(
            closes, ivr_arr, spread_width=spread_width,
            target_delta=target_delta, max_hold=max_hold,
            tp_pct=tp_pct, sl_pct=sl_pct, min_ivr=min_ivr, r=r,
        )
        return [t["pnl"] for t in results if "pnl" in t]
    except Exception:
        pass

    try:
        sim_spread_with_cost, _, CostModel = _import_full_validation()
        _synth_iv, _ = _import_synth_iv()
        cost = CostModel()
        rets = np.diff(np.log(closes))
        pnls = []
        i = 252
        while i < len(closes) - max_hold:
            window_vols = []
            for j in range(20, min(i, 252)):
                wv = float(np.std(rets[j - 20:j]) * np.sqrt(252))
                window_vols.append(wv)
            if not window_vols:
                i += 1
                continue
            current_vol = float(np.std(rets[max(0, i - 20):i]) * np.sqrt(252))
            iv_min = min(window_vols)
            iv_max = max(window_vols)
            ivr = ((current_vol - iv_min) / (iv_max - iv_min) * 100
                   if iv_max - iv_min > 0.001 else 50.0)
            if ivr < min_ivr:
                i += 1
                continue
            spot = closes[i]
            iv = _synth_iv(current_vol, max_hold)
            T = max_hold / 252
            res = sim_spread_with_cost(
                spot, iv, T, spread_width, target_delta, max_hold,
                tp_pct, sl_pct, r, closes, i, direction="BULL",
                cost_model=cost,
            )
            if res["pnl"] != 0:
                pnls.append(res["pnl"])
            i += max_hold + 1
        return pnls
    except Exception:
        return []


def _cs_pnl_for_cpcv(data, start, end, spread_width=5, target_delta=0.25,
                      max_hold=21, tp_pct=0.50, sl_pct=2.0, min_ivr=30):
    return _cs_backtest(data[start:end], spread_width=spread_width,
                        target_delta=target_delta, max_hold=max_hold,
                        tp_pct=tp_pct, sl_pct=sl_pct, min_ivr=min_ivr)


def _cs_pnl_for_noise(noisy_closes, spread_width=5, target_delta=0.25,
                       max_hold=21, tp_pct=0.50, sl_pct=2.0, min_ivr=30):
    return _cs_backtest(noisy_closes, spread_width=spread_width,
                        target_delta=target_delta, max_hold=max_hold,
                        tp_pct=tp_pct, sl_pct=sl_pct, min_ivr=min_ivr)


def _cs_eval_for_sensitivity(params, closes=None, r=0.05, **_extra):
    pnls = _cs_backtest(closes, **params, r=r)
    return _sharpe(pnls)


def _cs_stress(closes_slice, spread_width=5, target_delta=0.25,
               max_hold=21, tp_pct=0.50, sl_pct=2.0, min_ivr=30):
    return _cs_backtest(closes_slice, spread_width=spread_width,
                        target_delta=target_delta, max_hold=max_hold,
                        tp_pct=tp_pct, sl_pct=sl_pct, min_ivr=min_ivr)


def build_credit_spread_task() -> ValidationTask:
    defaults = dict(spread_width=5, target_delta=0.25, max_hold=21,
                    tp_pct=0.50, sl_pct=2.0, min_ivr=30)
    return ValidationTask(
        name="credit_spread",
        strategy_type="options",
        symbols=["SPY"],
        backtest_func=_cs_backtest,
        pnl_for_cpcv=_cs_pnl_for_cpcv,
        pnl_for_noise=_cs_pnl_for_noise,
        eval_for_sensitivity=_cs_eval_for_sensitivity,
        stress_func=_cs_stress,
        default_params=defaults,
        param_grid={
            "spread_width": [2, 3, 5, 7, 10],
            "target_delta": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "max_hold": [14, 21, 30, 45],
            "tp_pct": [0.30, 0.50, 0.75, 1.0],
            "sl_pct": [1.5, 2.0, 3.0],
        },
        economic_thesis=ECONOMIC_THESES["credit_spread"],
        needs_ivr=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  Builder: Straddle Squeeze
# ═══════════════════════════════════════════════════════════════════

def _straddle_backtest(closes, bb_lookback=126, bb_percentile_threshold=5,
                       max_holding_days=14, target_mult=1.5, r=0.05):
    """BB-squeeze straddle: buy ATM straddle when BB width percentile < threshold."""
    try:
        bs_price = _import_bs()
        bb_width_fn = _import_bb_width()
        _synth_iv, _hist_vol = _import_synth_iv()
    except Exception:
        return []

    if len(closes) < bb_lookback + 50:
        return []

    widths = bb_width_fn(closes)
    pnls = []
    i = bb_lookback + 20
    while i < len(closes) - max_holding_days:
        valid_w = widths[i - bb_lookback:i]
        valid_w = valid_w[~np.isnan(valid_w)]
        if len(valid_w) < bb_lookback // 2:
            i += 1
            continue
        current_w = widths[i]
        if np.isnan(current_w):
            i += 1
            continue
        threshold = float(np.percentile(valid_w, bb_percentile_threshold))
        if current_w > threshold:
            i += 1
            continue

        spot = closes[i]
        strike = round(spot)
        if spot <= 0:
            i += 1
            continue
        vol = _hist_vol(closes[:i + 1])
        T = max_holding_days / 252
        iv = _synth_iv(vol, max_holding_days)

        call_price = bs_price(spot, strike, T, r, iv, "CALL")
        put_price = bs_price(spot, strike, T, r, iv, "PUT")
        entry_cost = call_price + put_price
        if entry_cost <= 0.10:
            i += 1
            continue

        pnl = -entry_cost * 100
        for d in range(1, min(max_holding_days + 1, len(closes) - i)):
            future_spot = closes[i + d]
            T_rem = max((max_holding_days - d) / 252, 0.001)
            call_now = bs_price(future_spot, strike, T_rem, r, iv, "CALL")
            put_now = bs_price(future_spot, strike, T_rem, r, iv, "PUT")
            current_value = (call_now + put_now) * 100
            pnl = current_value - entry_cost * 100
            if current_value >= entry_cost * 100 * target_mult:
                break

        pnls.append(pnl)
        i += max_holding_days + 1

    return pnls


def _straddle_cpcv(data, start, end, bb_lookback=126,
                   bb_percentile_threshold=5, max_holding_days=14, target_mult=1.5):
    return _straddle_backtest(data[start:end], bb_lookback=bb_lookback,
                              bb_percentile_threshold=bb_percentile_threshold,
                              max_holding_days=max_holding_days,
                              target_mult=target_mult)


def _straddle_noise(noisy_closes, bb_lookback=126,
                    bb_percentile_threshold=5, max_holding_days=14, target_mult=1.5):
    return _straddle_backtest(noisy_closes, bb_lookback=bb_lookback,
                              bb_percentile_threshold=bb_percentile_threshold,
                              max_holding_days=max_holding_days,
                              target_mult=target_mult)


def _straddle_sensitivity(params, closes=None, r=0.05, **_extra):
    pnls = _straddle_backtest(closes, **params, r=r)
    return _sharpe(pnls)


def _straddle_stress(closes_slice, bb_lookback=126,
                     bb_percentile_threshold=5, max_holding_days=14, target_mult=1.5):
    return _straddle_backtest(closes_slice, bb_lookback=bb_lookback,
                              bb_percentile_threshold=bb_percentile_threshold,
                              max_holding_days=max_holding_days,
                              target_mult=target_mult)


def build_straddle_task() -> ValidationTask:
    defaults = dict(bb_lookback=126, bb_percentile_threshold=5,
                    max_holding_days=14, target_mult=1.5)
    return ValidationTask(
        name="straddle_squeeze",
        strategy_type="options",
        symbols=["SPY"],
        backtest_func=_straddle_backtest,
        pnl_for_cpcv=_straddle_cpcv,
        pnl_for_noise=_straddle_noise,
        eval_for_sensitivity=_straddle_sensitivity,
        stress_func=_straddle_stress,
        default_params=defaults,
        param_grid={
            "bb_lookback": [63, 126, 252],
            "bb_percentile_threshold": [3, 5, 8, 10],
            "max_holding_days": [7, 14, 21],
            "target_mult": [1.0, 1.5, 2.0],
        },
        economic_thesis=ECONOMIC_THESES["straddle_squeeze"],
        needs_ivr=False,
    )


# ═══════════════════════════════════════════════════════════════════
#  Builder: Earnings Spread
# ═══════════════════════════════════════════════════════════════════

def _earnings_backtest(closes, pre_event_days=5, post_event_days=1,
                       vol_threshold_mult=3.0, max_cost_pct=0.05, r=0.05):
    """Earnings straddle: heuristic event detection then straddle entry/exit."""
    try:
        bs_price = _import_bs()
        _synth_iv, _hist_vol = _import_synth_iv()
    except Exception:
        return []

    if len(closes) < 300:
        return []

    log_rets = np.diff(np.log(closes))
    pnls = []

    event_indices = []
    for i in range(20, len(log_rets)):
        rolling_std = float(np.std(log_rets[max(0, i - 20):i]))
        if rolling_std <= 0:
            continue
        if abs(log_rets[i]) > vol_threshold_mult * rolling_std:
            event_indices.append(i + 1)

    processed = set()
    for ev_idx in event_indices:
        entry_idx = ev_idx - pre_event_days
        exit_idx = ev_idx + post_event_days
        if entry_idx < 252 or exit_idx >= len(closes):
            continue
        bucket = ev_idx // 30
        if bucket in processed:
            continue
        processed.add(bucket)

        spot = closes[entry_idx]
        if spot <= 0:
            continue
        strike = round(spot)
        vol = _hist_vol(closes[:entry_idx + 1])
        T = (pre_event_days + post_event_days) / 252
        iv = _synth_iv(vol, pre_event_days + post_event_days)

        call_price = bs_price(spot, strike, T, r, iv, "CALL")
        put_price = bs_price(spot, strike, T, r, iv, "PUT")
        entry_cost = call_price + put_price
        if entry_cost <= 0.05:
            continue
        if entry_cost / spot > max_cost_pct:
            continue

        exit_spot = closes[exit_idx]
        T_rem = max(post_event_days / 252, 0.001)
        call_exit = bs_price(exit_spot, strike, T_rem, r, iv * 0.8, "CALL")
        put_exit = bs_price(exit_spot, strike, T_rem, r, iv * 0.8, "PUT")
        exit_value = call_exit + put_exit

        pnl = (exit_value - entry_cost) * 100
        pnls.append(pnl)

    return pnls


def _earnings_cpcv(data, start, end, pre_event_days=5, post_event_days=1,
                   vol_threshold_mult=3.0, max_cost_pct=0.05):
    return _earnings_backtest(data[start:end], pre_event_days=pre_event_days,
                              post_event_days=post_event_days,
                              vol_threshold_mult=vol_threshold_mult,
                              max_cost_pct=max_cost_pct)


def _earnings_noise(noisy_closes, pre_event_days=5, post_event_days=1,
                    vol_threshold_mult=3.0, max_cost_pct=0.05):
    return _earnings_backtest(noisy_closes, pre_event_days=pre_event_days,
                              post_event_days=post_event_days,
                              vol_threshold_mult=vol_threshold_mult,
                              max_cost_pct=max_cost_pct)


def _earnings_sensitivity(params, closes=None, r=0.05, **_extra):
    pnls = _earnings_backtest(closes, **params, r=r)
    return _sharpe(pnls)


def _earnings_stress(closes_slice, pre_event_days=5, post_event_days=1,
                     vol_threshold_mult=3.0, max_cost_pct=0.05):
    return _earnings_backtest(closes_slice, pre_event_days=pre_event_days,
                              post_event_days=post_event_days,
                              vol_threshold_mult=vol_threshold_mult,
                              max_cost_pct=max_cost_pct)


def build_earnings_task() -> ValidationTask:
    defaults = dict(pre_event_days=5, post_event_days=1,
                    vol_threshold_mult=3.0, max_cost_pct=0.05)
    return ValidationTask(
        name="earnings_spread",
        strategy_type="options",
        symbols=["SPY", "AAPL", "MSFT", "NVDA"],
        backtest_func=_earnings_backtest,
        pnl_for_cpcv=_earnings_cpcv,
        pnl_for_noise=_earnings_noise,
        eval_for_sensitivity=_earnings_sensitivity,
        stress_func=_earnings_stress,
        default_params=defaults,
        param_grid={
            "pre_event_days": [3, 5, 7],
            "vol_threshold_mult": [2.5, 3.0, 4.0],
            "max_cost_pct": [0.03, 0.05, 0.08],
        },
        economic_thesis=ECONOMIC_THESES["earnings_spread"],
        needs_ivr=False,
    )


# ═══════════════════════════════════════════════════════════════════
#  Builder: Wheel (CSP)
# ═══════════════════════════════════════════════════════════════════

def _wheel_backtest(closes, target_delta=0.25, dte=30, tp_pct=0.50,
                    min_ivr=30, r=0.05):
    """Wheel CSP simulation wrapper."""
    try:
        _, sim_wheel_with_cost, CostModel = _import_full_validation()
    except Exception:
        return []

    if len(closes) < 300:
        return []

    try:
        from backtest.turbo_core import precompute_ivr_fast
        ivr_arr = precompute_ivr_fast(closes)
    except Exception:
        rets = np.diff(np.log(closes))
        ivr_arr = np.full(len(closes), 50.0)
        for i in range(252, len(closes)):
            window_vols = []
            for j in range(20, min(i, 252)):
                wv = float(np.std(rets[j - 20:j]) * np.sqrt(252))
                window_vols.append(wv)
            if window_vols:
                current_vol = float(np.std(rets[max(0, i - 20):i]) * np.sqrt(252))
                iv_min, iv_max = min(window_vols), max(window_vols)
                ivr_arr[i] = ((current_vol - iv_min) / (iv_max - iv_min) * 100
                              if iv_max - iv_min > 0.001 else 50.0)

    trades = sim_wheel_with_cost(closes, ivr_arr, target_delta=target_delta,
                                 dte=dte, tp_pct=tp_pct, r=r,
                                 min_ivr=min_ivr)
    return [t["pnl"] for t in trades]


def _wheel_cpcv(data, start, end, target_delta=0.25, dte=30,
                tp_pct=0.50, min_ivr=30):
    return _wheel_backtest(data[start:end], target_delta=target_delta,
                           dte=dte, tp_pct=tp_pct, min_ivr=min_ivr)


def _wheel_noise(noisy_closes, target_delta=0.25, dte=30,
                 tp_pct=0.50, min_ivr=30):
    return _wheel_backtest(noisy_closes, target_delta=target_delta,
                           dte=dte, tp_pct=tp_pct, min_ivr=min_ivr)


def _wheel_sensitivity(params, closes=None, r=0.05, **_extra):
    pnls = _wheel_backtest(closes, **params, r=r)
    return _sharpe(pnls)


def _wheel_stress(closes_slice, target_delta=0.25, dte=30,
                  tp_pct=0.50, min_ivr=30):
    return _wheel_backtest(closes_slice, target_delta=target_delta,
                           dte=dte, tp_pct=tp_pct, min_ivr=min_ivr)


def build_wheel_task() -> ValidationTask:
    defaults = dict(target_delta=0.25, dte=30, tp_pct=0.50, min_ivr=30)
    return ValidationTask(
        name="wheel_csp",
        strategy_type="options",
        symbols=["SPY"],
        backtest_func=_wheel_backtest,
        pnl_for_cpcv=_wheel_cpcv,
        pnl_for_noise=_wheel_noise,
        eval_for_sensitivity=_wheel_sensitivity,
        stress_func=_wheel_stress,
        default_params=defaults,
        param_grid={
            "target_delta": [0.20, 0.25, 0.30, 0.35, 0.40],
            "dte": [14, 21, 30, 45],
            "tp_pct": [0.50, 0.75, 1.0],
            "min_ivr": [20, 30, 40],
        },
        economic_thesis=ECONOMIC_THESES["wheel_csp"],
        needs_ivr=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  Builder: Generic Equity Strategy
# ═══════════════════════════════════════════════════════════════════

_EQUITY_STRATEGY_CLASSES = {}

try:
    from strategy.momentum import MomentumStrategy
    _EQUITY_STRATEGY_CLASSES["momentum"] = MomentumStrategy
except Exception:
    pass

try:
    from strategy.mean_reversion import MeanReversionStrategy
    _EQUITY_STRATEGY_CLASSES["mean_reversion"] = MeanReversionStrategy
except Exception:
    pass

try:
    from strategy.breakout import BreakoutStrategy
    _EQUITY_STRATEGY_CLASSES["breakout"] = BreakoutStrategy
except Exception:
    pass

try:
    from strategy.rsi_reversal import RsiReversalStrategy
    _EQUITY_STRATEGY_CLASSES["rsi_reversal"] = RsiReversalStrategy
except Exception:
    pass

try:
    from strategy.multi_factor import MultiFactorStrategy
    _EQUITY_STRATEGY_CLASSES["multi_factor"] = MultiFactorStrategy
except Exception:
    pass

try:
    from backtest.backtester import Backtester as _Backtester
    _HAS_BACKTESTER = True
except Exception:
    _HAS_BACKTESTER = False


def _closes_to_ohlcv(closes):
    """Build a realistic OHLCV DataFrame from a close-price array.

    Synthesises open/high/low from close prices to give strategies
    meaningful bar ranges instead of zero-range bars.
    """
    c = np.asarray(closes, dtype=np.float64)
    n = len(c)
    opens = np.empty(n)
    opens[0] = c[0]
    opens[1:] = c[:-1]

    daily_range = np.abs(np.diff(c, prepend=c[0])) + c * 0.005
    highs = np.maximum(opens, c) + daily_range * 0.5
    lows = np.minimum(opens, c) - daily_range * 0.5
    lows = np.maximum(lows, c * 0.95)

    vol_base = 1e6
    vol_noise = np.random.default_rng(42).uniform(0.5, 2.0, n)
    volume = vol_base * vol_noise

    df = pd.DataFrame({
        "time_key": pd.date_range("2010-01-01", periods=n, freq="B")
                       .strftime("%Y-%m-%d"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": c,
        "volume": volume,
    })
    return df


# Cache for real DataFrames used by equity backtest
_EQUITY_DF_CACHE: dict[str, pd.DataFrame] = {}


def _equity_backtest(closes, strategy_name="momentum", strategy_class=None,
                     capital=3000.0, ohlcv_df=None, **strategy_params):
    """Generic equity strategy backtest returning list of trade PnLs."""
    if strategy_class is None:
        strategy_class = _EQUITY_STRATEGY_CLASSES.get(strategy_name)
    if strategy_class is None or not _HAS_BACKTESTER:
        return []

    if ohlcv_df is not None and not ohlcv_df.empty:
        df = ohlcv_df
    else:
        df = _closes_to_ohlcv(closes)
    try:
        try:
            strat = strategy_class(strategy_params)
        except TypeError:
            strat = strategy_class(strategy_name, strategy_params)
        bt = _Backtester(initial_capital=capital)
        result = bt.run(strat, "SIM", df, lookback=50, use_atr_stop=True)
        return [t["pnl"] for t in result.get("trades", []) if "pnl" in t]
    except Exception:
        return []


def _equity_cpcv(data, start, end, strategy_name="momentum",
                 strategy_class=None, **kw):
    return _equity_backtest(data[start:end], strategy_name=strategy_name,
                            strategy_class=strategy_class, **kw)


def _equity_noise(noisy_closes, strategy_name="momentum",
                  strategy_class=None, **kw):
    return _equity_backtest(noisy_closes, strategy_name=strategy_name,
                            strategy_class=strategy_class, **kw)


def _equity_sensitivity(params, closes=None, strategy_name="momentum",
                        strategy_class=None, **_extra):
    merged = {k: v for k, v in params.items()
              if k not in ("strategy_name", "strategy_class")}
    pnls = _equity_backtest(closes, strategy_name=strategy_name,
                            strategy_class=strategy_class, **merged)
    return _sharpe(pnls)


def _equity_stress(closes_slice, strategy_name="momentum",
                   strategy_class=None, **kw):
    return _equity_backtest(closes_slice, strategy_name=strategy_name,
                            strategy_class=strategy_class, **kw)


def build_equity_task(strategy_name: str, strategy_class=None) -> ValidationTask:
    if strategy_class is None:
        strategy_class = _EQUITY_STRATEGY_CLASSES.get(strategy_name)

    defaults = dict(strategy_name=strategy_name, strategy_class=strategy_class)

    def bt(closes, **kw):
        merged = {**defaults, **kw}
        return _equity_backtest(closes, **merged)

    def cpcv(data, start, end, **kw):
        merged = {**defaults, **kw}
        return _equity_cpcv(data, start, end, **merged)

    def noise(noisy_closes, **kw):
        merged = {**defaults, **kw}
        return _equity_noise(noisy_closes, **merged)

    def sens(params, closes=None):
        return _equity_sensitivity(params, closes=closes,
                                   strategy_name=strategy_name,
                                   strategy_class=strategy_class)

    def stress(closes_slice, **kw):
        merged = {**defaults, **kw}
        return _equity_stress(closes_slice, **merged)

    return ValidationTask(
        name=strategy_name,
        strategy_type="equity",
        symbols=["SPY"],
        backtest_func=bt,
        pnl_for_cpcv=cpcv,
        pnl_for_noise=noise,
        eval_for_sensitivity=sens,
        stress_func=stress,
        default_params=defaults,
        param_grid={},
        economic_thesis=ECONOMIC_THESES.get(strategy_name, ""),
        needs_ivr=False,
    )


# ═══════════════════════════════════════════════════════════════════
#  Builder: Momentum Rotation (ETF)
# ═══════════════════════════════════════════════════════════════════

try:
    from strategy.fractional.momentum_rotation import MomentumRotation as _MR
    _HAS_MR = True
except Exception:
    _HAS_MR = False


def _mr_backtest(closes, top_n=2, lookback=252, skip=21, sma_period=200,
                 budget=500.0, daily_data=None):
    """Momentum rotation backtest returning list of monthly PnL floats."""
    if not _HAS_MR:
        return []
    if daily_data is None:
        return []

    try:
        result = _MR.backtest_momentum(
            daily_data, budget=budget, top_n=top_n,
            lookback=lookback, skip=skip, sma_period=sma_period,
        )
        equity = result.get("equity", np.array([]))
        if len(equity) < 2:
            return []
        returns = np.diff(equity)
        return returns.tolist()
    except Exception:
        return []


def _mr_cpcv(data, start, end, top_n=2, lookback=252, skip=21,
             sma_period=200, budget=500.0, daily_data=None):
    return _mr_backtest(data[start:end] if daily_data is None else data,
                        top_n=top_n, lookback=lookback, skip=skip,
                        sma_period=sma_period, budget=budget,
                        daily_data=daily_data)


def _mr_noise(noisy_closes, top_n=2, lookback=252, skip=21,
              sma_period=200, budget=500.0, daily_data=None):
    return _mr_backtest(noisy_closes, top_n=top_n, lookback=lookback,
                        skip=skip, sma_period=sma_period, budget=budget,
                        daily_data=daily_data)


def _mr_sensitivity(params, closes=None, daily_data=None, budget=500.0, **_extra):
    pnls = _mr_backtest(closes, daily_data=daily_data, budget=budget, **params)
    return _sharpe(pnls)


def _mr_stress(closes_slice, top_n=2, lookback=252, skip=21,
               sma_period=200, budget=500.0, daily_data=None):
    return _mr_backtest(closes_slice, top_n=top_n, lookback=lookback,
                        skip=skip, sma_period=sma_period, budget=budget,
                        daily_data=daily_data)


def build_momentum_rotation_task() -> ValidationTask:
    defaults = dict(top_n=2, lookback=252, skip=21, sma_period=200, budget=500.0)
    return ValidationTask(
        name="momentum_rotation",
        strategy_type="etf",
        symbols=["SGOV", "BIL", "TLT", "VEA", "EEM", "XLF", "XLE", "IWM"],
        backtest_func=_mr_backtest,
        pnl_for_cpcv=_mr_cpcv,
        pnl_for_noise=_mr_noise,
        eval_for_sensitivity=_mr_sensitivity,
        stress_func=_mr_stress,
        default_params=defaults,
        param_grid={
            "top_n": [1, 2, 3],
            "lookback": [126, 189, 252],
            "skip": [0, 21, 42],
            "sma_period": [100, 150, 200, 252],
        },
        economic_thesis=ECONOMIC_THESES["momentum_rotation"],
        needs_ivr=False,
        capital=500.0,
    )


# ═══════════════════════════════════════════════════════════════════
#  Registry helpers
# ═══════════════════════════════════════════════════════════════════

def get_all_tasks() -> list[ValidationTask]:
    """Return all available validation tasks."""
    tasks = []

    tasks.append(build_credit_spread_task())
    tasks.append(build_straddle_task())
    tasks.append(build_earnings_task())
    tasks.append(build_wheel_task())

    for name in ("momentum", "mean_reversion", "breakout",
                 "rsi_reversal", "multi_factor"):
        cls = _EQUITY_STRATEGY_CLASSES.get(name)
        if cls is not None:
            tasks.append(build_equity_task(name, cls))

    tasks.append(build_momentum_rotation_task())

    return tasks


def get_task(name: str) -> Optional[ValidationTask]:
    """Return a specific task by name, or None."""
    for t in get_all_tasks():
        if t.name == name:
            return t
    return None
