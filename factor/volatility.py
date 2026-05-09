from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ivr(closes: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """IVR-like factor: percentile rank of current HV20 within trailing window."""
    returns = closes.pct_change()
    hv20 = returns.rolling(20).std() * np.sqrt(252)

    ivr = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for col in closes.columns:
        s = hv20[col]
        arr = s.values
        out = np.full(len(arr), np.nan)
        for i in range(window, len(arr)):
            current = arr[i]
            if np.isnan(current):
                continue
            hist = arr[i - window:i + 1]
            valid = hist[~np.isnan(hist)]
            if len(valid) < window // 2:
                continue
            out[i] = np.sum(valid < current) / len(valid)
        ivr[col] = out
    return ivr


def calc_hv_ratio(closes: pd.DataFrame, short_window: int = 20,
                  long_window: int = 60) -> pd.DataFrame:
    """HV ratio = HV_short / HV_long. Above 1 means vol expanding."""
    returns = closes.pct_change()
    hv_short = returns.rolling(short_window).std() * np.sqrt(252)
    hv_long = returns.rolling(long_window).std() * np.sqrt(252)
    return hv_short / hv_long.replace(0, np.nan)
