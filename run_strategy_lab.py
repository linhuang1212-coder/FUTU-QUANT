"""
$10K Strategy Lab — 运行全部 10 个策略回测 + 验证 + 报告

Usage:
  python run_strategy_lab.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from backtest.strategy_lab import (
    CAPITAL, YEARS, download_all, calc_metrics,
    strat_momentum, strat_gem, strat_mean_reversion,
    strat_trend_vol, strat_pairs, strat_aaa, strat_hrp,
    strat_factor_rotation, strat_xgboost, strat_ensemble,
    validate_strategy, generate_report,
)


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"  $10K STRATEGY LAB — {YEARS}-Year Backtest")
    print(f"  Capital: ${CAPITAL:,}")
    print("=" * 70)

    # ── Phase 1: Data ──────────────────────────────────────
    print("\n[Phase 1] Downloading data...")
    data = download_all()
    print(f"  Phase 1 done: {time.time() - t0:.0f}s\n")

    # ── Phase 2: Run all strategies ───────────────────────
    all_results = []

    def run(label, fn, *args, **kwargs):
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        try:
            result = fn(*args, **kwargs)
            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  CAGR={result.get('cagr', 0):.1%} | "
                      f"Sharpe={result.get('sharpe', 0):.2f} | "
                      f"MaxDD={result.get('max_drawdown', 0):.1%} | "
                      f"Final=${result.get('final_value', 0):,.0f}")
                all_results.append(result)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    # Strategy 1: ETF Momentum Rotation (multiple configs)
    etf_pool_data = {k: v for k, v in data.items()
                     if k in ["SGOV", "TLT", "XLK", "SMH", "XLF", "XLE",
                              "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE",
                              "VEA", "EEM", "SLV", "GDX", "XLB"]}
    run("Strategy 1a: Momentum Rotation (top5, lb252)",
        strat_momentum, etf_pool_data, 8000, 5, 252, 21)
    run("Strategy 1b: Momentum Rotation (top3, lb252)",
        strat_momentum, etf_pool_data, 8000, 3, 252, 21)

    # Strategy 2: Dual Momentum GEM
    run("Strategy 2: Dual Momentum GEM", strat_gem, data)

    # Strategy 3: Mean Reversion Z-Score
    run("Strategy 3a: Mean Reversion (SPY/QQQ/IWM)",
        strat_mean_reversion, data, 10000, 20, -2.0, 0.0)
    run("Strategy 3b: Mean Reversion (sector ETFs)",
        strat_mean_reversion, data, 10000, 20, -2.0, 0.0,
        ["XLK", "XLF", "XLE", "XLV", "SMH"])

    # Strategy 4: Trend Following + Vol Target
    run("Strategy 4: Trend Following + Vol Target",
        strat_trend_vol, data, 10000, 0.10)

    # Strategy 5: Pairs Trading
    run("Strategy 5: Pairs Trading", strat_pairs, data)

    # Strategy 6: Adaptive Asset Allocation
    run("Strategy 6: Adaptive Asset Allocation",
        strat_aaa, data, 10000, 5, 126)

    # Strategy 7: HRP + alternatives
    run("Strategy 7a: HRP Risk Parity", strat_hrp, data, 10000, "hrp")
    run("Strategy 7b: Min Variance", strat_hrp, data, 10000, "min_variance")
    run("Strategy 7c: Inv-Vol Risk Parity", strat_hrp, data, 10000, "risk_parity")
    run("Strategy 7d: Equal Weight", strat_hrp, data, 10000, "equal")

    # Strategy 8: Factor ETF Rotation
    run("Strategy 8: Factor ETF Rotation",
        strat_factor_rotation, data, 10000, 63)

    # Strategy 9: XGBoost ML
    run("Strategy 9: XGBoost ML Selection",
        strat_xgboost, data, 10000, 504, 21, 5)

    print(f"\n  Phase 2 done: {time.time() - t0:.0f}s")
    print(f"  {len(all_results)} strategies completed successfully\n")

    # Strategy 10: Ensemble
    print(f"\n{'─' * 60}")
    print(f"  Strategy 10: Multi-Strategy Ensemble")
    print(f"{'─' * 60}")
    ensemble_results = strat_ensemble(all_results, CAPITAL, 3)
    for er in ensemble_results:
        print(f"  {er['name']}: Sharpe={er.get('sharpe', 0):.2f} | "
              f"CAGR={er.get('cagr', 0):.1%}")
        all_results.append(er)

    # ── Phase 3: Validation ───────────────────────────────
    print(f"\n{'=' * 70}")
    print("  [Phase 3] Validation — Top 5 strategies")
    print(f"{'=' * 70}")

    ranked = sorted(all_results, key=lambda x: x.get("sharpe", -999), reverse=True)
    validations = []
    for r in ranked[:7]:
        if "equity" in r:
            print(f"  Validating: {r['name']}...")
            v = validate_strategy(r["equity"], r["name"], n_mc=500, n_folds=6)
            validations.append(v)
            kf = f"{v.get('kfold_pass', 0)}/{v.get('kfold_total', 6)}"
            robust = "ROBUST" if v.get("kfold_robust") else "WEAK"
            print(f"    Sharpe={v.get('actual_sharpe', 0):.2f} | "
                  f"K-Fold={kf} ({robust}) | "
                  f"PBO={v.get('pbo_probability', 0):.1%} | "
                  f"Recovery={v.get('recovery_days', 0)}d")

    print(f"\n  Phase 3 done: {time.time() - t0:.0f}s\n")

    # ── Phase 4: Report ───────────────────────────────────
    print(f"{'=' * 70}")
    print("  [Phase 4] Generating report")
    print(f"{'=' * 70}")

    report_results = [{k: v for k, v in r.items() if k != "equity"}
                      for r in all_results]
    report_path = str(Path(__file__).resolve().parent / "docs" / "strategy_lab_report.md")
    generate_report(report_results, validations, report_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  {len(all_results)} strategies evaluated")
    print(f"  Report: {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
