from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.logger import setup_logger

logger = setup_logger("factor.quintile")


def quintile_backtest(factor_panel: pd.DataFrame, returns_panel: pd.DataFrame,
                      n_groups: int = 5,
                      holding_period: int = 5) -> dict:
    """Factor quintile backtest.

    Sorts symbols into groups by factor value, computes equal-weight returns.
    Returns dict with group_cum_returns, group_stats, long_short_cum, monotonicity.
    """
    forward_returns = returns_panel.rolling(holding_period).sum().shift(-holding_period)
    rebalance_dates = factor_panel.index[::holding_period]

    group_period_returns: dict[int, list[float]] = {i: [] for i in range(1, n_groups + 1)}
    group_dates: list = []

    for date in rebalance_dates:
        if date not in forward_returns.index:
            continue
        factor_vals = factor_panel.loc[date].dropna()
        fwd_ret = forward_returns.loc[date]
        common = factor_vals.index.intersection(fwd_ret.dropna().index)
        if len(common) < n_groups * 2:
            continue

        ranks = factor_vals[common].rank(pct=True)
        group_dates.append(date)

        for g in range(1, n_groups + 1):
            lower = (g - 1) / n_groups
            upper = g / n_groups
            if g == 1:
                members = ranks[ranks <= upper].index
            else:
                members = ranks[(ranks > lower) & (ranks <= upper)].index

            avg_ret = fwd_ret[members].mean() if len(members) > 0 else 0.0
            group_period_returns[g].append(avg_ret)

    if not group_dates:
        return {"group_cum_returns": pd.DataFrame(), "group_stats": pd.DataFrame(),
                "long_short_cum": pd.Series(dtype=float), "monotonicity": 0.0}

    group_ret_df = pd.DataFrame(group_period_returns, index=group_dates)
    group_ret_df.columns = [f"G{i}" for i in range(1, n_groups + 1)]

    group_cum = (1 + group_ret_df).cumprod()

    ls = group_ret_df[f"G{n_groups}"] - group_ret_df["G1"]
    ls_cum = (1 + ls).cumprod()

    periods_per_year = 252 / holding_period
    stats_rows = {}
    for col in group_ret_df.columns:
        s = group_ret_df[col]
        ann_ret = s.mean() * periods_per_year
        ann_vol = s.std() * np.sqrt(periods_per_year)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        stats_rows[col] = {"ann_return": round(ann_ret, 4),
                           "ann_vol": round(ann_vol, 4),
                           "sharpe": round(sharpe, 3)}

    ls_ann_ret = ls.mean() * periods_per_year
    ls_ann_vol = ls.std() * np.sqrt(periods_per_year)
    ls_sharpe = ls_ann_ret / ls_ann_vol if ls_ann_vol > 0 else 0
    stats_rows["L-S"] = {"ann_return": round(ls_ann_ret, 4),
                         "ann_vol": round(ls_ann_vol, 4),
                         "sharpe": round(ls_sharpe, 3)}

    group_stats = pd.DataFrame(stats_rows).T

    group_means = [group_ret_df[f"G{i}"].mean() for i in range(1, n_groups + 1)]
    mono, _ = spearmanr(list(range(1, n_groups + 1)), group_means)

    return {
        "group_cum_returns": group_cum,
        "group_stats": group_stats,
        "long_short_cum": ls_cum,
        "monotonicity": round(mono, 3) if not np.isnan(mono) else 0.0,
    }


def print_quintile_report(results: dict, factor_name: str = ""):
    """Pretty-print quintile backtest results."""
    header = f"Quintile Backtest: {factor_name}" if factor_name else "Quintile Backtest"
    print("=" * 60)
    print(header)
    print("=" * 60)
    if results["group_stats"].empty:
        print("  No results (insufficient data)")
        return
    print(results["group_stats"].to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Monotonicity: {results['monotonicity']:.3f}")
    print("  (1.0 = perfect ascending, -1.0 = perfect descending)")
