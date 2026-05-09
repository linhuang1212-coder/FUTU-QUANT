from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardization across each cross-section (row)."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank standardization to [0, 1] across each cross-section."""
    return panel.rank(axis=1, pct=True)


def industry_neutralize(panel: pd.DataFrame,
                        industry_map: dict[str, str]) -> pd.DataFrame:
    """Remove industry-level mean from each symbol's factor value."""
    result = panel.copy()
    industries = set(industry_map.values())
    for date in panel.index:
        row = panel.loc[date]
        for ind in industries:
            members = [s for s in row.index if industry_map.get(s) == ind]
            if len(members) > 1:
                ind_mean = row[members].mean()
                result.loc[date, members] = row[members] - ind_mean
    return result


def winsorize(panel: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """Clip extreme values beyond n_std from cross-sectional mean."""
    result = panel.copy()
    for date in panel.index:
        row = panel.loc[date]
        valid = row.dropna()
        if len(valid) < 3:
            continue
        mu, sigma = valid.mean(), valid.std()
        if sigma == 0:
            continue
        result.loc[date] = row.clip(mu - n_std * sigma, mu + n_std * sigma)
    return result
