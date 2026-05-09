"""
Portfolio Weight Optimizer — HRP + 备选方案

核心方法:
  1. HRP (Hierarchical Risk Parity) — 基于协方差矩阵的层次风险平价
  2. 等风险贡献 (Risk Parity) — 各标的风险贡献相等
  3. 等权 (Equal Weight) — fallback

参考: PyPortfolioOpt (5.6k stars, GitHub)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("optimizer")


def compute_weights_hrp(symbols: list[str],
                        period: str = "1y") -> dict[str, float]:
    """Compute HRP weights using historical return covariance.

    Downloads price data via yfinance, computes covariance,
    then applies Hierarchical Risk Parity.

    Returns:
        {symbol: weight} dict, weights sum to ~1.0
    """
    returns = _fetch_returns(symbols, period)
    if returns is None or returns.empty or len(returns.columns) < 2:
        logger.warning("[HRP] Insufficient data, falling back to equal weight")
        return _equal_weight(symbols)

    try:
        from pypfopt import HRPOpt

        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        cleaned = hrp.clean_weights(cutoff=0.01)

        result = {}
        for sym in symbols:
            result[sym] = cleaned.get(sym, 0.0)

        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        non_zero = sum(1 for v in result.values() if v > 0.01)
        logger.info(f"[HRP] {non_zero}/{len(symbols)} stocks with weight > 1%")

        top3 = sorted(result.items(), key=lambda x: -x[1])[:3]
        for sym, w in top3:
            logger.info(f"  {sym:6s}: {w:.1%}")

        return result

    except Exception as e:
        logger.warning(f"[HRP] Optimization failed ({e}), falling back to equal weight")
        return _equal_weight(symbols)


def compute_weights_min_variance(symbols: list[str],
                                 period: str = "1y") -> dict[str, float]:
    """Minimum variance portfolio — lowest overall portfolio volatility."""
    returns = _fetch_returns(symbols, period)
    if returns is None or returns.empty or len(returns.columns) < 2:
        return _equal_weight(symbols)

    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns

        mu = expected_returns.mean_historical_return(returns)
        S = risk_models.sample_cov(returns)
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        cleaned = ef.clean_weights(cutoff=0.01)

        result = {sym: cleaned.get(sym, 0.0) for sym in symbols}
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    except Exception as e:
        logger.warning(f"[MinVar] Failed ({e}), using equal weight")
        return _equal_weight(symbols)


def compute_weights_risk_parity(symbols: list[str],
                                period: str = "1y") -> dict[str, float]:
    """Inverse-volatility risk parity — simple but effective."""
    returns = _fetch_returns(symbols, period)
    if returns is None or returns.empty:
        return _equal_weight(symbols)

    vols = returns.std()
    vols = vols.replace(0, np.nan).dropna()

    if vols.empty:
        return _equal_weight(symbols)

    inv_vol = 1.0 / vols
    total = inv_vol.sum()
    weights = (inv_vol / total).to_dict()

    result = {}
    for sym in symbols:
        result[sym] = weights.get(sym, 0.0)

    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return result


def compute_weights(symbols: list[str],
                    method: str = "hrp",
                    period: str = "1y") -> dict[str, float]:
    """Unified interface for weight computation.

    Args:
        symbols: list of ticker symbols
        method: 'hrp', 'min_variance', 'risk_parity', or 'equal'
        period: yfinance period string
    """
    dispatch = {
        "hrp": compute_weights_hrp,
        "min_variance": compute_weights_min_variance,
        "risk_parity": compute_weights_risk_parity,
        "equal": _equal_weight,
    }
    fn = dispatch.get(method, compute_weights_hrp)
    if method == "equal":
        return fn(symbols)
    return fn(symbols, period=period)


def _equal_weight(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


def _fetch_returns(symbols: list[str],
                   period: str = "1y") -> Optional[pd.DataFrame]:
    """Download and compute daily returns for a list of symbols."""
    if not symbols or len(symbols) < 2:
        return None
    try:
        import yfinance as yf

        tickers = " ".join(symbols)
        data = yf.download(tickers, period=period, progress=False)
        if data.empty:
            return None

        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=symbols[0])

        # Drop columns with too many NaN
        valid = close.dropna(axis=1, thresh=int(len(close) * 0.8))
        if valid.empty or len(valid.columns) < 2:
            return None

        returns = valid.pct_change().dropna()
        return returns

    except Exception as e:
        logger.error(f"[Optimizer] Price fetch failed: {e}")
        return None
