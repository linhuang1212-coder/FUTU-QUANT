"""
Parallel execution engine for the 7-Gate validation pipeline.

Gate-level execution is SEQUENTIAL (Gate 3 -> 4 -> 5 -> 6) for determinism.
Within each gate, independent work items are parallelized via joblib (loky backend).
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from typing import Callable, Optional

from joblib import Parallel, delayed

from backtest.full_validation import (
    cpcv_splits, CPCVResult, MonteCarloResult,
    SensitivityResult, FullSensitivityResult,
    StressPeriodResult, StressTestResult, STRESS_PERIODS,
    _max_drawdown, build_validation, FullValidationResult,
)
from backtest.strategy_adapters import ValidationTask, _sharpe


def _default_n_jobs(n_jobs: Optional[int] = None) -> int:
    if n_jobs is not None:
        return n_jobs
    return min(16, max(1, (os.cpu_count() or 4) - 2))


# ═══════════════════════════════════════════════════════════════════
#  Workers — module-level for pickling
# ═══════════════════════════════════════════════════════════════════

def _cpcv_worker(split, data, pnl_func, **kwargs):
    test_idx = split["test_idx"]
    if len(test_idx) < 50:
        return None
    test_start = test_idx[0]
    test_end = test_idx[-1] + 1
    pnls = pnl_func(data, test_start, test_end, **kwargs)
    if len(pnls) < 3:
        return 0.0
    pnl_arr = np.array(pnls)
    std = np.std(pnl_arr)
    return float(np.mean(pnl_arr) / std * np.sqrt(12)) if std > 0 else 0.0


def _noise_worker(seed, closes, noise_pct, strategy_func, **kwargs):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_pct, size=len(closes))
    noisy = closes * (1 + noise)
    noisy = np.maximum(noisy, 0.01)
    pnls = strategy_func(noisy, **kwargs)
    if len(pnls) < 3:
        return 0.0
    arr = np.array(pnls)
    std = np.std(arr)
    return float(np.mean(arr) / std * np.sqrt(12)) if std > 0 else 0.0


def _sens_worker(param_name, value, base_params, eval_func, **kwargs):
    params = {**base_params, param_name: value}
    return (param_name, value, eval_func(params, **kwargs))


def _stress_worker(name, start, end, df, strategy_func, lookback, **kwargs):
    end_mask = df["time_key"] <= end
    start_mask = df["time_key"] >= start
    stress_idx = df.index[start_mask & end_mask]
    if len(stress_idx) < 5:
        return None

    first_stress_idx = stress_idx[0]
    expanded_start = max(0, first_stress_idx - lookback)
    slice_df = df.iloc[expanded_start:stress_idx[-1] + 1]
    if len(slice_df) < 50:
        return None

    closes = slice_df["close"].values
    pnls = strategy_func(closes, **kwargs)

    period = StressPeriodResult(name=name, start_date=start, end_date=end)
    if pnls:
        pnl_arr = np.array(pnls)
        period.n_trades = len(pnls)
        period.total_pnl = float(np.sum(pnl_arr))
        period.max_drawdown = _max_drawdown(pnl_arr)
    return period


# ═══════════════════════════════════════════════════════════════════
#  Gate 3: CPCV — parallel across splits
# ═══════════════════════════════════════════════════════════════════

def run_cpcv_parallel(
    pnl_func: Callable,
    data: np.ndarray,
    n_groups: int = 10,
    k_test: int = 2,
    purge_bars: int = 5,
    embargo_bars: int = 2,
    n_jobs: Optional[int] = None,
    **strategy_kwargs,
) -> CPCVResult:
    n_jobs = _default_n_jobs(n_jobs)
    splits = cpcv_splits(len(data), n_groups, k_test, purge_bars, embargo_bars)
    result = CPCVResult(n_paths=len(splits))

    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_cpcv_worker)(split, data, pnl_func, **strategy_kwargs)
        for split in splits
    )

    result.oos_sharpes = [s for s in raw if s is not None]

    if result.oos_sharpes:
        neg = sum(1 for s in result.oos_sharpes if s <= 0)
        result.pbo = neg / len(result.oos_sharpes)
        result.avg_oos_sharpe = float(np.mean(result.oos_sharpes))
        result.median_oos_sharpe = float(np.median(result.oos_sharpes))

    result.pass_gate = result.pbo < 0.40
    return result


# ═══════════════════════════════════════════════════════════════════
#  Gate 4: Monte Carlo — noise injection parallelized
# ═══════════════════════════════════════════════════════════════════

def run_monte_carlo_parallel(
    pnls: np.ndarray,
    closes: Optional[np.ndarray] = None,
    strategy_func: Optional[Callable] = None,
    n_shuffle: int = 1000,
    n_bootstrap: int = 1000,
    n_noise: int = 200,
    noise_pct: float = 0.005,
    n_jobs: Optional[int] = None,
    **strategy_kwargs,
) -> MonteCarloResult:
    n_jobs = _default_n_jobs(n_jobs)
    mc = MonteCarloResult()

    # --- Shuffle (sequential — O(n) per iter, fast enough) ---
    original_dd = _max_drawdown(pnls)
    rng = np.random.default_rng(42)
    dd_dist = np.zeros(n_shuffle)
    for i in range(n_shuffle):
        dd_dist[i] = _max_drawdown(rng.permutation(pnls))

    mc.shuffle_max_dd_original = original_dd
    mc.shuffle_max_dd_median = float(np.median(dd_dist))
    mc.shuffle_max_dd_percentile = float(np.mean(dd_dist <= original_dd) * 100)

    # --- Bootstrap (sequential — same rationale) ---
    n = len(pnls)
    if n >= 10:
        rng_b = np.random.default_rng(42)
        sharpes = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng_b.choice(pnls, size=n, replace=True)
            std = np.std(sample)
            sharpes[i] = float(np.mean(sample) / std * np.sqrt(12)) if std > 0 else 0.0
        mc.bootstrap_sharpe_mean = float(np.mean(sharpes))
        mc.bootstrap_sharpe_ci_lo = float(np.percentile(sharpes, 2.5))
        mc.bootstrap_sharpe_ci_hi = float(np.percentile(sharpes, 97.5))
        mc.bootstrap_pass = mc.bootstrap_sharpe_ci_lo > 0
    else:
        mc.bootstrap_pass = False

    # --- Noise injection (parallelized) ---
    if closes is not None and strategy_func is not None:
        noise_sharpes = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_noise_worker)(42 + i, closes, noise_pct, strategy_func, **strategy_kwargs)
            for i in range(n_noise)
        )
        ns = np.array(noise_sharpes, dtype=float)
        mc.noise_sharpe_median = float(np.median(ns))
        mc.noise_sharpe_ci_lo = float(np.percentile(ns, 5))
        mc.noise_pass = mc.noise_sharpe_median > 0
    else:
        mc.noise_pass = True

    mc.pass_gate = mc.bootstrap_pass and mc.noise_pass
    return mc


# ═══════════════════════════════════════════════════════════════════
#  Gate 5: Sensitivity — parallel across all (param, value) pairs
# ═══════════════════════════════════════════════════════════════════

def run_sensitivity_parallel(
    param_grid: dict,
    eval_func: Callable,
    base_params: dict,
    n_jobs: Optional[int] = None,
    **fixed_kwargs,
) -> FullSensitivityResult:
    n_jobs = _default_n_jobs(n_jobs)

    flat_jobs = []
    for param_name, values in param_grid.items():
        for val in values:
            flat_jobs.append((param_name, val))

    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_sens_worker)(pname, val, base_params, eval_func, **fixed_kwargs)
        for pname, val in flat_jobs
    )

    grouped: dict[str, list] = {}
    for pname, val, sharpe in raw:
        grouped.setdefault(pname, []).append((val, sharpe))

    fs = FullSensitivityResult()
    for param_name, values in param_grid.items():
        pairs = grouped.get(param_name, [])
        vals_sorted = sorted(pairs, key=lambda x: (
            x[0] if isinstance(x[0], (int, float)) else 0
        ))
        sr = SensitivityResult(
            param_name=param_name,
            values=[v for v, _ in vals_sorted],
            sharpes=[s for _, s in vals_sorted],
        )

        sharpes_arr = np.array(sr.sharpes)
        valid = sharpes_arr[np.isfinite(sharpes_arr) & (sharpes_arr > -100)]

        if len(valid) >= 3:
            best = np.max(valid)
            if best > 0:
                within_15pct = np.sum(valid >= best * 0.85)
                sr.sensitivity_score = float(within_15pct / len(valid))
                sr.is_plateau = sr.sensitivity_score >= 0.60

                for j in range(len(valid) - 1):
                    if valid[j] > 0 and (valid[j] - valid[j + 1]) / valid[j] > 0.50:
                        sr.is_cliff = True
                        break

        fs.param_results.append(sr)

    fs.compute()
    return fs


# ═══════════════════════════════════════════════════════════════════
#  Gate 6: Stress test — parallel across 4 windows
# ═══════════════════════════════════════════════════════════════════

def run_stress_parallel(
    df: pd.DataFrame,
    strategy_func: Callable,
    capital: float = 3000,
    lookback: int = 260,
    n_jobs: Optional[int] = None,
    **strategy_kwargs,
) -> StressTestResult:
    n_jobs = min(4, _default_n_jobs(n_jobs))

    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_stress_worker)(name, start, end, df, strategy_func, lookback, **strategy_kwargs)
        for name, (start, end) in STRESS_PERIODS.items()
    )

    result = StressTestResult()
    result.periods = [p for p in raw if p is not None]
    result.compute(capital=capital)
    return result


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator: validate one strategy
# ═══════════════════════════════════════════════════════════════════

def validate_strategy(
    task: ValidationTask,
    cache=None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> FullValidationResult:
    t0 = time.time()
    n_jobs = _default_n_jobs(n_jobs)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Validating: {task.name} ({task.strategy_type})")
        print(f"{'='*60}")

    # ── Load data ──
    closes = np.array([])
    ivr_arr = None
    df = pd.DataFrame()

    if cache:
        closes_dict = {sym: cache.get_closes(sym) for sym in task.symbols}
        primary_sym = task.symbols[0] if task.symbols else ""
        closes = closes_dict.get(primary_sym.replace("US.", ""), np.array([]))
        if task.needs_ivr:
            ivr_arr = cache.get_ivr(primary_sym)
        df = cache.get_frame(primary_sym)
    else:
        try:
            from data.downloader import load_daily
            if task.symbols:
                primary_sym = task.symbols[0]
                df = load_daily(primary_sym)
                if not df.empty and "close" in df.columns:
                    closes = df["close"].values
        except Exception:
            pass

    # ── Gate 2: Base backtest ──
    if verbose:
        print(f"  Gate 2: Base backtest...", end=" ", flush=True)
    t_gate = time.time()
    base_pnls = []
    if task.backtest_func and len(closes) > 0:
        try:
            extra_kw = dict(task.default_params)
            if task.strategy_type == "equity" and not df.empty:
                extra_kw["ohlcv_df"] = df
            base_pnls = task.backtest_func(closes, **extra_kw)
            if base_pnls is None:
                base_pnls = []
        except Exception:
            base_pnls = []

    cost_sharpe = _sharpe(base_pnls) if base_pnls else 0.0
    n_trades = len(base_pnls)
    win_rate = float(np.mean(np.array(base_pnls) > 0)) if base_pnls else 0.0
    if verbose:
        print(f"done ({time.time() - t_gate:.1f}s, {n_trades} trades, Sharpe={cost_sharpe:.2f})")

    # ── Gate 3: CPCV ──
    if verbose:
        print(f"  Gate 3: CPCV ({n_jobs} workers)...", end=" ", flush=True)
    t_gate = time.time()
    cpcv_result = None
    if task.pnl_for_cpcv and len(closes) > 500:
        try:
            cpcv_result = run_cpcv_parallel(
                task.pnl_for_cpcv, closes, n_jobs=n_jobs, **task.default_params,
            )
        except Exception as e:
            if verbose:
                print(f"[error: {e}]", end=" ")
    if verbose:
        elapsed_g = time.time() - t_gate
        pbo_str = f"PBO={cpcv_result.pbo:.1%}" if cpcv_result else "skipped"
        print(f"done ({elapsed_g:.1f}s, {pbo_str})")

    # ── Gate 4: Monte Carlo ──
    if verbose:
        print(f"  Gate 4: Monte Carlo...", end=" ", flush=True)
    t_gate = time.time()
    mc_result = None
    if len(base_pnls) >= 5:
        try:
            mc_result = run_monte_carlo_parallel(
                np.array(base_pnls),
                closes=closes if task.pnl_for_noise else None,
                strategy_func=task.pnl_for_noise,
                n_jobs=n_jobs,
                **task.default_params,
            )
        except Exception as e:
            if verbose:
                print(f"[error: {e}]", end=" ")
    if verbose:
        elapsed_g = time.time() - t_gate
        mc_str = f"boot_ci_lo={mc_result.bootstrap_sharpe_ci_lo:.2f}" if mc_result else "skipped"
        print(f"done ({elapsed_g:.1f}s, {mc_str})")

    # ── Gate 5: Sensitivity ──
    n_dims = len(task.param_grid) if task.param_grid else 0
    if verbose:
        print(f"  Gate 5: Sensitivity ({n_dims} dims)...", end=" ", flush=True)
    t_gate = time.time()
    sens_result = None
    if task.eval_for_sensitivity and task.param_grid:
        try:
            extra = {}
            if closes is not None and len(closes) > 0:
                extra["closes"] = closes
            if ivr_arr is not None:
                extra["ivr_arr"] = ivr_arr
            sens_result = run_sensitivity_parallel(
                task.param_grid, task.eval_for_sensitivity,
                task.default_params, n_jobs=n_jobs, **extra,
            )
        except Exception as e:
            if verbose:
                print(f"[error: {e}]", end=" ")
    if verbose:
        elapsed_g = time.time() - t_gate
        sens_str = f"score={sens_result.overall_score:.2f}" if sens_result else "skipped"
        print(f"done ({elapsed_g:.1f}s, {sens_str})")

    # ── Gate 6: Stress test ──
    if verbose:
        print(f"  Gate 6: Stress test...", end=" ", flush=True)
    t_gate = time.time()
    stress_result = None
    if task.stress_func and not df.empty and "time_key" in df.columns:
        try:
            stress_result = run_stress_parallel(
                df, task.stress_func, capital=task.capital,
                n_jobs=n_jobs, **task.default_params,
            )
        except Exception as e:
            if verbose:
                print(f"[error: {e}]", end=" ")
    if verbose:
        elapsed_g = time.time() - t_gate
        stress_str = "pass" if (stress_result and stress_result.pass_gate) else "skipped/fail"
        print(f"done ({elapsed_g:.1f}s, {stress_str})")

    # ── Gate 7: DSR ──
    if verbose:
        print(f"  Gate 7: DSR...", end=" ", flush=True)
    dsr_dict = None
    try:
        from backtest.validation import deflated_sharpe_ratio
        if len(base_pnls) >= 10 and cost_sharpe > 0:
            dsr_dict = deflated_sharpe_ratio(
                cost_sharpe, len(base_pnls),
                n_trials=max(1, len(task.param_grid)),
            )
    except Exception:
        pass
    if verbose:
        dsr_str = f"DSR={dsr_dict.get('deflated_sharpe', 0):.3f}" if dsr_dict else "skipped"
        print(f"done ({dsr_str})")

    # ── Build result ──
    result = build_validation(
        task.name, cost_sharpe,
        cpcv_result, mc_result, sens_result, stress_result, dsr_dict,
        n_trades=n_trades,
        win_rate=win_rate,
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Completed in {elapsed:.1f}s  |  "
              f"Score: {result.total_score:.0f}/{result.max_possible}  |  {result.verdict}")

    return result


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator: validate all tasks sequentially
# ═══════════════════════════════════════════════════════════════════

def validate_all(
    tasks: list[ValidationTask],
    cache=None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> list[FullValidationResult]:
    results: list[FullValidationResult] = []
    for task in tasks:
        try:
            result = validate_strategy(task, cache=cache, n_jobs=n_jobs, verbose=verbose)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {task.name} failed: {e}")
    return results
