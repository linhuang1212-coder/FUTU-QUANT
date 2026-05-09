from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ep_factor(fundamentals_df: pd.DataFrame) -> pd.Series:
    """Earnings yield = 1/PE. Higher value = cheaper stock."""
    if "pe" not in fundamentals_df.columns:
        return pd.Series(dtype=float, name="EP")
    pe = fundamentals_df["pe"].replace(0, np.nan)
    return (1.0 / pe).rename("EP")


def calc_roe_factor(fundamentals_df: pd.DataFrame) -> pd.Series:
    """ROE quality factor."""
    if "roe" not in fundamentals_df.columns:
        return pd.Series(dtype=float, name="ROE")
    return fundamentals_df["roe"].rename("ROE")


def calc_revenue_growth(fundamentals_df: pd.DataFrame) -> pd.Series:
    """Revenue growth factor."""
    if "revenue_growth" not in fundamentals_df.columns:
        return pd.Series(dtype=float, name="REV_GROWTH")
    return fundamentals_df["revenue_growth"].rename("REV_GROWTH")


def build_all_fundamental(fundamentals_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build all available fundamental factors."""
    result = {}
    for name, func in [("EP", calc_ep_factor), ("ROE", calc_roe_factor),
                        ("REV_GROWTH", calc_revenue_growth)]:
        s = func(fundamentals_df)
        if s.notna().any():
            result[name] = s
    return result
