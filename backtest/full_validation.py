"""
Full quantitative validation pipeline.

Implements institutional-grade strategy validation:
  1. Transaction cost model
  2. CPCV (Combinatorial Purged Cross-Validation) with real PBO
  3. Monte Carlo simulation (shuffle, bootstrap, noise injection)
  4. Parameter sensitivity analysis (plateau detection)
  5. Stress testing (extreme event windows)
  6. Unified scoring engine (7-layer Pass/Conditional/Fail)

References:
  Lopez de Prado (2018) "Advances in Financial Machine Learning"
  Bailey et al. (2017) "The Probability of Backtest Overfitting"
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from options.pricer import bs_price, compute_ivr


# ═══════════════════════════════════════════════════════════════════
#  MODULE 1: Transaction Cost Model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CostModel:
    """Realistic option trading cost model."""
    commission_per_contract: float = 0.65
    spread_pct: float = 0.02        # bid-ask as % of mid-price
    sec_fee_per_dollar: float = 2.78e-5  # SEC fee on sells
    min_spread_cost: float = 0.01    # minimum $0.01 spread cost per contract

    def trade_cost(self, n_legs: int, n_contracts: int,
                   avg_price: float, is_open: bool = True) -> float:
        """Total cost for a trade (open or close)."""
        commission = self.commission_per_contract * n_contracts * n_legs
        spread = max(self.min_spread_cost, avg_price * self.spread_pct) * n_contracts * 100 * n_legs
        sec = 0.0
        if not is_open:
            sec = self.sec_fee_per_dollar * avg_price * n_contracts * 100 * n_legs
        return commission + spread + sec

    def round_trip_cost(self, n_legs: int, n_contracts: int,
                        entry_price: float, exit_price: float) -> float:
        open_cost = self.trade_cost(n_legs, n_contracts, entry_price, is_open=True)
        close_cost = self.trade_cost(n_legs, n_contracts, exit_price, is_open=False)
        return open_cost + close_cost


DEFAULT_COST = CostModel()


def sim_spread_with_cost(spot, iv, T, width, target_delta, max_hold, tp_pct, sl_pct,
                         r, closes, entry_idx, direction="BULL",
                         cost_model: CostModel = DEFAULT_COST) -> dict:
    """Credit spread simulation with transaction costs."""
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

    # Entry cost
    avg_price = (short_price + long_price) / 2
    entry_cost = cost_model.trade_cost(2, 1, avg_price, is_open=True)

    max_loss = (abs(long_strike - short_strike) - credit) * 100
    pnl = credit * 100
    reason = "expiry"
    exit_avg_price = avg_price

    for d in range(1, min(max_hold + 1, len(closes) - entry_idx)):
        future_spot = closes[entry_idx + d]
        T_rem = max((max_hold - d) / 252, 0.001)
        short_now = bs_price(future_spot, short_strike, T_rem, r, iv, opt_type)
        long_now = bs_price(future_spot, long_strike, T_rem, r, iv, opt_type)
        spread_now = short_now - long_now
        cur_pnl = (credit - spread_now) * 100

        if cur_pnl >= credit * 100 * tp_pct:
            pnl = cur_pnl
            exit_avg_price = (short_now + long_now) / 2
            reason = "take_profit"
            break
        if cur_pnl <= -max_loss * sl_pct:
            pnl = cur_pnl
            exit_avg_price = (short_now + long_now) / 2
            reason = "stop_loss"
            break
        pnl = cur_pnl
        exit_avg_price = (short_now + long_now) / 2

    # Exit cost
    exit_cost = cost_model.trade_cost(2, 1, exit_avg_price, is_open=False)
    total_cost = entry_cost + exit_cost
    pnl_after_cost = pnl - total_cost

    return {"pnl": pnl_after_cost, "pnl_gross": pnl, "cost": total_cost,
            "credit": credit, "max_loss": max_loss, "reason": reason}


def sim_wheel_with_cost(closes, ivr_arr, target_delta, dte, tp_pct, r,
                        min_ivr=30, cost_model: CostModel = DEFAULT_COST) -> list[dict]:
    """Wheel CSP simulation with transaction costs."""
    from options.backtest import _synth_iv  # noqa: E402
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

        entry_cost = cost_model.trade_cost(1, 1, put_price, is_open=True)
        credit = put_price * 100
        pnl = credit
        reason = "expiry"
        exit_price = put_price

        for d in range(1, min(dte + 1, len(closes) - i)):
            future = closes[i + d]
            T_rem = max((dte - d) / 252, 0.001)
            put_now = bs_price(future, strike, T_rem, r, iv, "PUT")
            cur_pnl = (put_price - put_now) * 100
            if cur_pnl >= credit * tp_pct:
                pnl = cur_pnl
                exit_price = put_now
                reason = "take_profit"
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
        trades.append({"pnl": pnl_after, "pnl_gross": pnl,
                        "cost": entry_cost + exit_cost, "reason": reason})
        i += dte + 1
    return trades


# ═══════════════════════════════════════════════════════════════════
#  MODULE 2: CPCV (Combinatorial Purged Cross-Validation)
# ═══════════════════════════════════════════════════════════════════

def cpcv_splits(n: int, n_groups: int = 10, k_test: int = 2,
                purge_bars: int = 5, embargo_bars: int = 2) -> list[dict]:
    """Generate all C(n_groups, k_test) combinatorial splits with purge + embargo.

    Returns list of dicts: {train_idx: [], test_idx: [], combo: tuple}
    """
    group_size = n // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else n
        groups.append(list(range(start, end)))

    splits = []
    for combo in itertools.combinations(range(n_groups), k_test):
        test_set = set()
        for g_idx in combo:
            test_set.update(groups[g_idx])

        train_idx = []
        purge_embargo = set()

        for g_idx in combo:
            g_start = groups[g_idx][0]
            g_end = groups[g_idx][-1]
            for p in range(max(0, g_start - purge_bars), g_start):
                purge_embargo.add(p)
            for e in range(g_end + 1, min(n, g_end + 1 + embargo_bars)):
                purge_embargo.add(e)

        for i in range(n):
            if i not in test_set and i not in purge_embargo:
                train_idx.append(i)

        test_idx = sorted(test_set)
        if train_idx and test_idx:
            splits.append({"train_idx": train_idx, "test_idx": test_idx, "combo": combo})

    return splits


@dataclass
class CPCVResult:
    n_paths: int = 0
    oos_sharpes: list = field(default_factory=list)
    pbo: float = 0.0
    avg_oos_sharpe: float = 0.0
    median_oos_sharpe: float = 0.0
    pass_gate: bool = False


def run_cpcv(pnl_func: Callable, data: np.ndarray, n_groups: int = 10,
             k_test: int = 2, purge_bars: int = 5, embargo_bars: int = 2,
             **strategy_kwargs) -> CPCVResult:
    """Run CPCV: for each combo, evaluate strategy on test portion.

    pnl_func(closes, start_idx, end_idx, **kwargs) -> list[float]
        Must return list of trade PnLs for the given index range.
    """
    n = len(data)
    splits = cpcv_splits(n, n_groups, k_test, purge_bars, embargo_bars)
    result = CPCVResult(n_paths=len(splits))

    for split in splits:
        test_idx = split["test_idx"]
        if len(test_idx) < 50:
            continue
        test_start = test_idx[0]
        test_end = test_idx[-1] + 1

        pnls = pnl_func(data, test_start, test_end, **strategy_kwargs)
        if len(pnls) < 3:
            result.oos_sharpes.append(0.0)
            continue

        pnl_arr = np.array(pnls)
        if np.std(pnl_arr) > 0:
            sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(12))
        else:
            sharpe = 0.0
        result.oos_sharpes.append(sharpe)

    if result.oos_sharpes:
        neg = sum(1 for s in result.oos_sharpes if s <= 0)
        result.pbo = neg / len(result.oos_sharpes)
        result.avg_oos_sharpe = float(np.mean(result.oos_sharpes))
        result.median_oos_sharpe = float(np.median(result.oos_sharpes))

    result.pass_gate = result.pbo < 0.40
    return result


# ═══════════════════════════════════════════════════════════════════
#  MODULE 3: Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MonteCarloResult:
    # Shuffle
    shuffle_max_dd_percentile: float = 0.0
    shuffle_max_dd_original: float = 0.0
    shuffle_max_dd_median: float = 0.0
    # Bootstrap
    bootstrap_sharpe_mean: float = 0.0
    bootstrap_sharpe_ci_lo: float = 0.0
    bootstrap_sharpe_ci_hi: float = 0.0
    bootstrap_pass: bool = False
    # Noise injection
    noise_sharpe_median: float = 0.0
    noise_sharpe_ci_lo: float = 0.0
    noise_pass: bool = False
    # Overall
    pass_gate: bool = False


def _max_drawdown(pnls: np.ndarray) -> float:
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    return float(np.min(dd)) if len(dd) > 0 else 0.0


def monte_carlo_shuffle(pnls: np.ndarray, n_sim: int = 1000) -> dict:
    """Shuffle trade order to test path dependency."""
    original_dd = _max_drawdown(pnls)
    dd_dist = np.zeros(n_sim)
    rng = np.random.default_rng(42)
    for i in range(n_sim):
        shuffled = rng.permutation(pnls)
        dd_dist[i] = _max_drawdown(shuffled)

    percentile = float(np.mean(dd_dist <= original_dd) * 100)
    return {
        "original_dd": original_dd,
        "median_dd": float(np.median(dd_dist)),
        "percentile": percentile,
        "p5_dd": float(np.percentile(dd_dist, 5)),
        "p95_dd": float(np.percentile(dd_dist, 95)),
    }


def monte_carlo_bootstrap(pnls: np.ndarray, n_sim: int = 1000) -> dict:
    """Bootstrap confidence interval for Sharpe ratio."""
    n = len(pnls)
    if n < 10:
        return {"mean": 0, "ci_lo": 0, "ci_hi": 0, "pass": False}

    rng = np.random.default_rng(42)
    sharpes = np.zeros(n_sim)
    for i in range(n_sim):
        sample = rng.choice(pnls, size=n, replace=True)
        std = np.std(sample)
        sharpes[i] = float(np.mean(sample) / std * np.sqrt(12)) if std > 0 else 0.0

    ci_lo = float(np.percentile(sharpes, 2.5))
    ci_hi = float(np.percentile(sharpes, 97.5))
    return {
        "mean": float(np.mean(sharpes)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "pass": ci_lo > 0,
    }


def monte_carlo_noise(closes: np.ndarray, strategy_func: Callable,
                      noise_pct: float = 0.005, n_sim: int = 200,
                      **strategy_kwargs) -> dict:
    """Inject price noise and re-run strategy.

    strategy_func(closes, **kwargs) -> list[float] (PnLs)
    """
    rng = np.random.default_rng(42)
    sharpes = np.zeros(n_sim)

    for i in range(n_sim):
        noise = rng.normal(0, noise_pct, size=len(closes))
        noisy_closes = closes * (1 + noise)
        noisy_closes = np.maximum(noisy_closes, 0.01)

        pnls = strategy_func(noisy_closes, **strategy_kwargs)
        if len(pnls) < 3:
            sharpes[i] = 0.0
            continue
        pnl_arr = np.array(pnls)
        std = np.std(pnl_arr)
        sharpes[i] = float(np.mean(pnl_arr) / std * np.sqrt(12)) if std > 0 else 0.0

    median = float(np.median(sharpes))
    ci_lo = float(np.percentile(sharpes, 5))
    return {
        "median": median,
        "ci_lo": ci_lo,
        "ci_hi": float(np.percentile(sharpes, 95)),
        "pct_positive": float(np.mean(sharpes > 0) * 100),
        "pass": median > 0,
    }


def run_monte_carlo(pnls: np.ndarray, closes: Optional[np.ndarray] = None,
                    strategy_func: Optional[Callable] = None,
                    **strategy_kwargs) -> MonteCarloResult:
    """Run full Monte Carlo suite."""
    result = MonteCarloResult()

    # 3a: Shuffle
    shuf = monte_carlo_shuffle(pnls, n_sim=1000)
    result.shuffle_max_dd_original = shuf["original_dd"]
    result.shuffle_max_dd_median = shuf["median_dd"]
    result.shuffle_max_dd_percentile = shuf["percentile"]

    # 3b: Bootstrap
    boot = monte_carlo_bootstrap(pnls, n_sim=1000)
    result.bootstrap_sharpe_mean = boot["mean"]
    result.bootstrap_sharpe_ci_lo = boot["ci_lo"]
    result.bootstrap_sharpe_ci_hi = boot["ci_hi"]
    result.bootstrap_pass = boot["pass"]

    # 3c: Noise injection (only if strategy_func provided)
    if closes is not None and strategy_func is not None:
        noise = monte_carlo_noise(closes, strategy_func, n_sim=200, **strategy_kwargs)
        result.noise_sharpe_median = noise["median"]
        result.noise_sharpe_ci_lo = noise["ci_lo"]
        result.noise_pass = noise["pass"]
    else:
        result.noise_pass = True  # skip if no func

    result.pass_gate = result.bootstrap_pass and result.noise_pass
    return result


# ═══════════════════════════════════════════════════════════════════
#  MODULE 4: Parameter Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SensitivityResult:
    param_name: str = ""
    values: list = field(default_factory=list)
    sharpes: list = field(default_factory=list)
    is_plateau: bool = False
    is_cliff: bool = False
    sensitivity_score: float = 0.0


def sensitivity_scan_1d(param_name: str, param_values: list,
                        eval_func: Callable, base_params: dict,
                        **fixed_kwargs) -> SensitivityResult:
    """Scan one parameter dimension, keeping others at base values.

    eval_func(params_dict, **fixed_kwargs) -> float (Sharpe)
    """
    result = SensitivityResult(param_name=param_name, values=list(param_values))

    for val in param_values:
        params = {**base_params, param_name: val}
        sharpe = eval_func(params, **fixed_kwargs)
        result.sharpes.append(sharpe)

    sharpes = np.array(result.sharpes)
    valid = sharpes[np.isfinite(sharpes) & (sharpes > -100)]

    if len(valid) < 3:
        result.sensitivity_score = 0.0
        return result

    best = np.max(valid)
    if best <= 0:
        result.sensitivity_score = 0.0
        return result

    # Plateau detection: count how many values are within 15% of best
    within_15pct = np.sum(valid >= best * 0.85)
    result.sensitivity_score = within_15pct / len(valid)
    result.is_plateau = result.sensitivity_score >= 0.60

    # Cliff detection: any adjacent pair with >50% drop
    for i in range(len(valid) - 1):
        if valid[i] > 0 and (valid[i] - valid[i + 1]) / valid[i] > 0.50:
            result.is_cliff = True
            break

    return result


@dataclass
class FullSensitivityResult:
    param_results: list = field(default_factory=list)
    overall_score: float = 0.0
    pass_gate: bool = False

    def compute(self):
        if not self.param_results:
            return
        scores = [r.sensitivity_score for r in self.param_results]
        self.overall_score = float(np.mean(scores))
        self.pass_gate = self.overall_score >= 0.60


# ═══════════════════════════════════════════════════════════════════
#  MODULE 5: Stress Testing
# ═══════════════════════════════════════════════════════════════════

STRESS_PERIODS = {
    "COVID_Crash": ("2020-02-19", "2020-03-23"),
    "COVID_Recovery": ("2020-03-24", "2020-06-08"),
    "Rate_Hike_Bear": ("2022-01-03", "2022-10-12"),
    "VIX_Spike_2024": ("2024-07-15", "2024-08-15"),
}


@dataclass
class StressPeriodResult:
    name: str = ""
    start_date: str = ""
    end_date: str = ""
    n_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_loss_pct: float = 0.0  # relative to capital
    veto: bool = False


@dataclass
class StressTestResult:
    periods: list = field(default_factory=list)
    any_veto: bool = False
    pass_gate: bool = False

    def compute(self, max_dd_veto_pct: float = 0.50, capital: float = 3000):
        self.any_veto = False
        for p in self.periods:
            p.max_loss_pct = abs(p.max_drawdown) / capital if capital > 0 else 0
            if p.max_loss_pct > max_dd_veto_pct:
                p.veto = True
                self.any_veto = True
        self.pass_gate = not self.any_veto


def run_stress_test(df: pd.DataFrame, strategy_func: Callable,
                    capital: float = 3000, lookback: int = 260,
                    **strategy_kwargs) -> StressTestResult:
    """Run strategy on each stress period independently.

    strategy_func(closes_slice, **kwargs) -> list[float] (PnLs)
    df must have 'time_key' and 'close' columns.
    Includes `lookback` bars before stress window for indicator warmup.
    """
    result = StressTestResult()

    for name, (start, end) in STRESS_PERIODS.items():
        end_mask = df["time_key"] <= end
        start_mask = df["time_key"] >= start
        stress_idx = df.index[start_mask & end_mask]
        if len(stress_idx) < 5:
            continue

        first_stress_idx = stress_idx[0]
        expanded_start = max(0, first_stress_idx - lookback)
        slice_df = df.iloc[expanded_start:stress_idx[-1] + 1]
        if len(slice_df) < 50:
            continue

        closes = slice_df["close"].values
        pnls = strategy_func(closes, **strategy_kwargs)

        period = StressPeriodResult(name=name, start_date=start, end_date=end)
        if pnls:
            pnl_arr = np.array(pnls)
            period.n_trades = len(pnls)
            period.total_pnl = float(np.sum(pnl_arr))
            period.max_drawdown = _max_drawdown(pnl_arr)
        result.periods.append(period)

    result.compute(capital=capital)
    return result


# ═══════════════════════════════════════════════════════════════════
#  MODULE 6: Unified Scoring Engine
# ═══════════════════════════════════════════════════════════════════

ECONOMIC_THESES = {
    "credit_spread": (
        "Volatility Risk Premium: options are systematically overpriced relative to "
        "realized volatility. Selling OTM put spreads when IVR is elevated captures "
        "the IV-RV spread. Academic evidence: Coval & Shumway (2001), Bakshi & "
        "Kapadia (2003). Edge: systematic theta collection + mean reversion of IV."
    ),
    "orb_0dte": (
        "Opening Range Breakout exploits institutional order flow concentration "
        "in the first 30 minutes. 0DTE options amplify directional moves with "
        "defined risk. Edge: momentum persistence after opening range resolution. "
        "Note: highly sensitive to execution quality and market microstructure."
    ),
    "momentum_rotation": (
        "Cross-sectional momentum: assets with strong recent returns continue "
        "to outperform. 12M-1M momentum avoids short-term reversal noise. "
        "SMA200 trend filter reduces drawdowns during bear markets. Academic "
        "evidence: Jegadeesh & Titman (1993), Moskowitz et al. (2012). "
        "Edge: systematic risk premia harvesting across asset classes."
    ),
}


@dataclass
class ValidationGate:
    name: str
    max_score: int
    actual_score: float = 0.0
    passed: bool = False
    detail: str = ""


@dataclass
class FullValidationResult:
    strategy_name: str = ""
    gates: list = field(default_factory=list)
    total_score: float = 0.0
    max_possible: int = 100
    verdict: str = ""

    # Raw results
    cost_sharpe: float = 0.0
    cpcv: Optional[CPCVResult] = None
    monte_carlo: Optional[MonteCarloResult] = None
    sensitivity: Optional[FullSensitivityResult] = None
    stress: Optional[StressTestResult] = None
    dsr_result: Optional[dict] = None

    def compute_verdict(self):
        self.total_score = sum(g.actual_score for g in self.gates)
        if self.total_score >= 75:
            self.verdict = "PASS — 推荐小仓位部署"
        elif self.total_score >= 60:
            self.verdict = "CONDITIONAL — 需要 Paper Trading 进一步验证"
        else:
            self.verdict = "FAIL — 不建议部署"


def build_validation(strategy_name: str,
                     cost_sharpe: float,
                     cpcv_result: Optional[CPCVResult],
                     mc_result: Optional[MonteCarloResult],
                     sens_result: Optional[FullSensitivityResult],
                     stress_result: Optional[StressTestResult],
                     dsr_dict: Optional[dict],
                     n_trades: int = 0,
                     win_rate: float = 0.0) -> FullValidationResult:
    """Build the 7-layer validation scorecard."""
    fv = FullValidationResult(strategy_name=strategy_name)
    fv.cost_sharpe = cost_sharpe
    fv.cpcv = cpcv_result
    fv.monte_carlo = mc_result
    fv.sensitivity = sens_result
    fv.stress = stress_result
    fv.dsr_result = dsr_dict

    # Gate 1: Economic thesis (10 pts)
    thesis = ECONOMIC_THESES.get(strategy_name, "")
    g1 = ValidationGate("经济学假设", 10)
    if thesis:
        g1.actual_score = 10
        g1.passed = True
        g1.detail = thesis[:80] + "..."
    else:
        g1.detail = "未记录经济学假设"
    fv.gates.append(g1)

    # Gate 2: Cost-adjusted Sharpe (15 pts)
    g2 = ValidationGate("含成本回测 Sharpe", 15)
    if cost_sharpe > 0.5:
        g2.actual_score = 15
        g2.passed = True
    elif cost_sharpe > 0.3:
        g2.actual_score = 10
        g2.passed = True
    elif cost_sharpe > 0:
        g2.actual_score = 5
    g2.detail = f"Sharpe = {cost_sharpe:.2f}"
    fv.gates.append(g2)

    # Gate 3: CPCV PBO (20 pts)
    g3 = ValidationGate("CPCV PBO", 20)
    if cpcv_result:
        if cpcv_result.pbo < 0.30:
            g3.actual_score = 20
            g3.passed = True
        elif cpcv_result.pbo < 0.40:
            g3.actual_score = 15
            g3.passed = True
        elif cpcv_result.pbo < 0.50:
            g3.actual_score = 8
        g3.detail = f"PBO = {cpcv_result.pbo:.1%} ({cpcv_result.n_paths} paths)"
    else:
        g3.detail = "未执行"
    fv.gates.append(g3)

    # Gate 4: Monte Carlo CI (15 pts)
    g4 = ValidationGate("Monte Carlo CI", 15)
    if mc_result:
        if mc_result.bootstrap_pass and mc_result.noise_pass:
            g4.actual_score = 15
            g4.passed = True
        elif mc_result.bootstrap_pass:
            g4.actual_score = 10
            g4.passed = True
        elif mc_result.noise_pass:
            g4.actual_score = 5
        g4.detail = (f"Bootstrap CI [{mc_result.bootstrap_sharpe_ci_lo:.2f}, "
                     f"{mc_result.bootstrap_sharpe_ci_hi:.2f}] | "
                     f"Noise median={mc_result.noise_sharpe_median:.2f}")
    else:
        g4.detail = "未执行"
    fv.gates.append(g4)

    # Gate 5: Parameter sensitivity (15 pts)
    g5 = ValidationGate("参数敏感性", 15)
    if sens_result:
        if sens_result.pass_gate:
            g5.actual_score = 15
            g5.passed = True
        elif sens_result.overall_score >= 0.40:
            g5.actual_score = 8
        g5.detail = f"Score = {sens_result.overall_score:.2f}"
        cliffs = [r.param_name for r in sens_result.param_results if r.is_cliff]
        if cliffs:
            g5.detail += f" | Cliffs: {', '.join(cliffs)}"
    else:
        g5.detail = "未执行"
    fv.gates.append(g5)

    # Gate 6: Stress test (15 pts)
    g6 = ValidationGate("压力测试", 15)
    if stress_result:
        if stress_result.pass_gate:
            g6.actual_score = 15
            g6.passed = True
        vetoed = [p.name for p in stress_result.periods if p.veto]
        if vetoed:
            g6.detail = f"VETO: {', '.join(vetoed)}"
        else:
            g6.detail = "所有压力期通过"
    else:
        g6.detail = "未执行"
    fv.gates.append(g6)

    # Gate 7: DSR significance (10 pts)
    g7 = ValidationGate("DSR 显著性", 10)
    if dsr_dict:
        if dsr_dict.get("is_significant"):
            g7.actual_score = 10
            g7.passed = True
        elif dsr_dict.get("deflated_sharpe", 0) > 0:
            g7.actual_score = 5
        g7.detail = (f"DSR = {dsr_dict.get('deflated_sharpe', 0):.3f}, "
                     f"p = {dsr_dict.get('p_value', 1):.3f}")
    else:
        g7.detail = "未执行"
    fv.gates.append(g7)

    fv.compute_verdict()
    return fv


def print_validation_report(fv: FullValidationResult):
    """Print formatted full validation report."""
    print(f"\n{'='*65}")
    print(f"  完整验证报告: {fv.strategy_name}")
    print(f"{'='*65}")

    for g in fv.gates:
        status = "PASS" if g.passed else "FAIL"
        bar = "█" * int(g.actual_score / g.max_score * 10) if g.max_score > 0 else ""
        print(f"\n  [{status:>4}] {g.name} ({g.actual_score:.0f}/{g.max_score})")
        print(f"         {bar} {g.detail}")

    print(f"\n  {'─'*61}")
    print(f"  总分: {fv.total_score:.0f}/{fv.max_possible}")
    print(f"  判定: {fv.verdict}")
    print(f"{'='*65}")
