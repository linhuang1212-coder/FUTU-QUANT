from __future__ import annotations
import math
import numpy as np
from typing import Optional


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "CALL") -> float:
    """Black-Scholes option price. T in years."""
    if T <= 0 or sigma <= 0:
        if option_type == "CALL":
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "CALL":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                option_type: str = "CALL", tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
    """Newton-Raphson IV solver."""
    if T <= 0 or price <= 0:
        return None

    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, sigma, option_type)
        vega = S * _norm_pdf(
            (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        ) * math.sqrt(T)
        if vega < 1e-12:
            break
        sigma -= (p - price) / vega
        if sigma <= 0:
            sigma = 0.01
        if abs(p - price) < tol:
            return sigma
    return sigma if abs(bs_price(S, K, T, r, sigma, option_type) - price) < tol * 10 else None


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = "CALL") -> dict:
    """Compute Delta, Gamma, Theta, Vega."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "CALL" else max(K - S, 0)
        delta = 1.0 if intrinsic > 0 and option_type == "CALL" else (-1.0 if intrinsic > 0 else 0.0)
        return {"delta": delta, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)

    if option_type == "CALL":
        delta = _norm_cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
    else:
        delta = _norm_cdf(d1) - 1
        theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) / 100  # per 1% move

    return {"delta": round(delta, 4), "gamma": round(gamma, 6),
            "theta": round(theta, 4), "vega": round(vega, 4)}


def compute_ivr(current_iv: float, hist_iv: list[float]) -> float:
    """IV Rank: where current IV sits in past year's range (0-100)."""
    if not hist_iv or len(hist_iv) < 10:
        return 50.0
    iv_min = min(hist_iv)
    iv_max = max(hist_iv)
    if iv_max - iv_min < 0.001:
        return 50.0
    return round((current_iv - iv_min) / (iv_max - iv_min) * 100, 1)


def compute_ivp(current_iv: float, hist_iv: list[float]) -> float:
    """IV Percentile: % of days IV was below current level."""
    if not hist_iv:
        return 50.0
    below = sum(1 for iv in hist_iv if iv < current_iv)
    return round(below / len(hist_iv) * 100, 1)


def bb_width(closes: np.ndarray, period: int = 20, std_mult: float = 2.0) -> np.ndarray:
    """Bollinger Band Width = (upper - lower) / middle."""
    if len(closes) < period:
        return np.full(len(closes), np.nan)
    import pandas as pd
    s = pd.Series(closes)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return width.values
