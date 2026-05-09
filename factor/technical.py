from __future__ import annotations

import numpy as np
import pandas as pd


def calc_momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Momentum = P_t / P_{t-window} - 1."""
    return prices.pct_change(window)


def calc_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def calc_turnover(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Relative turnover: short-term average / long-term average."""
    avg_vol = volume.rolling(window).mean()
    long_avg = volume.rolling(window * 5).mean()
    return avg_vol / long_avg.replace(0, np.nan)


def calc_reversal(returns: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Short-term reversal: negative of recent cumulative return."""
    return -returns.rolling(window).sum()


def build_all_technical(prices: pd.DataFrame,
                        volumes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all standard technical factors at once."""
    returns = prices.pct_change()
    return {
        "MOM_1M": calc_momentum(prices, 21),
        "MOM_3M": calc_momentum(prices, 63),
        "MOM_6M": calc_momentum(prices, 126),
        "MOM_12M": calc_momentum(prices, 252),
        "VOL_20D": calc_volatility(returns, 20),
        "VOL_60D": calc_volatility(returns, 60),
        "TURNOVER": calc_turnover(volumes, 20),
        "REVERSAL": calc_reversal(returns, 5),
    }
