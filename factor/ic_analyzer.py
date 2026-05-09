from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.logger import setup_logger

logger = setup_logger("factor.ic_analyzer")


def calc_ic_series(factor_panel: pd.DataFrame, returns_panel: pd.DataFrame,
                   method: str = "spearman", lag: int = 5) -> pd.Series:
    """Calculate factor IC time series.

    IC_t = corr(factor_t, forward_return_{t+lag})
    """
    forward_returns = returns_panel.rolling(lag).sum().shift(-lag)
    common_dates = factor_panel.index.intersection(forward_returns.index)

    ic_list = []
    for date in common_dates:
        f = factor_panel.loc[date]
        r = forward_returns.loc[date]
        valid = f.notna() & r.notna()
        if valid.sum() < 5:
            ic_list.append(np.nan)
            continue

        if method == "spearman":
            corr, _ = sp_stats.spearmanr(f[valid], r[valid])
        else:
            corr, _ = sp_stats.pearsonr(f[valid], r[valid])
        ic_list.append(corr)

    return pd.Series(ic_list, index=common_dates, name="IC")


def ic_summary(ic: pd.Series) -> dict:
    """Comprehensive IC statistics."""
    ic_clean = ic.dropna()
    if len(ic_clean) < 2:
        return {"IC_mean": np.nan, "IC_std": np.nan, "IC_IR": np.nan,
                "IC_pos_ratio": np.nan, "IC_abs_gt002": np.nan, "t_stat": np.nan}

    ic_mean = ic_clean.mean()
    ic_std = ic_clean.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    t_stat = ic_mean / (ic_std / np.sqrt(len(ic_clean))) if ic_std > 0 else 0.0

    return {
        "IC_mean": round(ic_mean, 4),
        "IC_std": round(ic_std, 4),
        "IC_IR": round(ic_ir, 4),
        "IC_pos_ratio": round((ic_clean > 0).mean(), 4),
        "IC_abs_gt002": round((ic_clean.abs() > 0.02).mean(), 4),
        "t_stat": round(t_stat, 4),
    }


def ic_report(factors: dict[str, pd.DataFrame], returns: pd.DataFrame,
              lag: int = 5,
              method: str = "spearman") -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Batch IC analysis for all factors.

    Returns (summary_table sorted by IC_IR, dict of IC series).
    """
    ic_series_dict = {}
    summaries = {}

    for name, panel in factors.items():
        ic = calc_ic_series(panel, returns, method=method, lag=lag)
        ic_series_dict[name] = ic
        summaries[name] = ic_summary(ic)
        logger.info(f"[IC] {name}: mean={summaries[name]['IC_mean']:.4f}, "
                    f"IR={summaries[name]['IC_IR']:.3f}")

    table = pd.DataFrame(summaries).T
    table.index.name = "Factor"
    table = table.sort_values("IC_IR", ascending=False)
    return table, ic_series_dict


def print_ic_report(table: pd.DataFrame, lag: int = 5):
    """Pretty-print IC analysis summary."""
    print("=" * 70)
    print(f"Factor IC Analysis ({lag}-day forward returns, Rank IC)")
    print("=" * 70)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("IC_IR > 0.5 is generally considered a good factor")
    print("|t_stat| > 2 means IC is statistically significant")
