"""Volatility Target Management System.

Provides VIX filtering, ADX regime detection, dynamic position sizing,
and drawdown governance for leveraged ETF trading.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketRegime:
    """Current market regime assessment."""
    vix_level: float
    adx_value: float
    is_trending: bool       # ADX > threshold
    vix_ok: bool            # VIX < entry threshold
    vix_danger: bool        # VIX > force-close threshold
    position_scale: float   # 0.0 - 1.0, how much of normal position to take
    regime_label: str       # "bull_trend", "bull_range", "high_vol", "danger"


class VolatilityTargetManager:
    """Manages position sizing and market regime detection."""

    def __init__(
        self,
        vix_entry_max: float = 28.0,
        vix_force_close: float = 35.0,
        vix_reduce_threshold: float = 22.0,
        adx_trend_threshold: float = 25.0,
        adx_period: int = 14,
        vol_target: float = 0.18,
        ewma_lambda: float = 0.94,
        dd_threshold: float = -0.20,
        dd_scale_factor: float = 0.5,
        max_position_scale: float = 0.95,
    ):
        self.vix_entry_max = vix_entry_max
        self.vix_force_close = vix_force_close
        self.vix_reduce_threshold = vix_reduce_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_period = adx_period
        self.vol_target = vol_target
        self.ewma_lambda = ewma_lambda
        self.dd_threshold = dd_threshold
        self.dd_scale_factor = dd_scale_factor
        self.max_position_scale = max_position_scale

    def get_vix_level(self, quote_ctx) -> Optional[float]:
        """Fetch current VIX from Futu. Returns None if unavailable."""
        try:
            from futu import RET_OK
            # Try US.VIX (CBOE VIX index on Futu)
            for vix_code in ["US.VIX", "US.VIXM"]:
                ret, data = quote_ctx.get_market_snapshot([vix_code])
                if ret == RET_OK and data is not None and len(data) > 0:
                    return float(data.iloc[0]["last_price"])
        except Exception:
            pass
        return None

    def compute_adx(self, df: pd.DataFrame, period: Optional[int] = None) -> float:
        """Compute ADX from daily OHLC data. Returns latest ADX value."""
        p = period or self.adx_period
        if len(df) < p * 2:
            return 0.0

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        plus_dm = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0),
            0.0
        )
        minus_dm = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0),
            0.0
        )

        def wilder_smooth(arr, period):
            result = np.empty_like(arr)
            result[:period] = np.nan
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - result[i-1] / period + arr[i]
            return result

        atr = wilder_smooth(tr, p)
        plus_di = 100 * wilder_smooth(plus_dm, p) / np.where(atr > 0, atr, 1)
        minus_di = 100 * wilder_smooth(minus_dm, p) / np.where(atr > 0, atr, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1)
        adx = wilder_smooth(dx[~np.isnan(dx)], p)

        valid = adx[~np.isnan(adx)]
        return float(valid[-1]) if len(valid) > 0 else 0.0

    def compute_ewma_vol(self, df: pd.DataFrame) -> float:
        """Compute EWMA annualized volatility from daily closes."""
        if len(df) < 20:
            return 0.3  # conservative default
        rets = df["close"].pct_change().dropna().values
        var = 0.0
        lam = self.ewma_lambda
        for r in rets:
            var = lam * var + (1 - lam) * r * r
        return float(np.sqrt(var) * np.sqrt(252))

    def compute_drawdown(self, df: pd.DataFrame) -> float:
        """Compute current drawdown from peak (as negative fraction)."""
        if len(df) < 2:
            return 0.0
        prices = df["close"].values
        peak = np.maximum.accumulate(prices)
        dd = (prices[-1] / peak[-1]) - 1.0
        return float(dd)

    def assess_regime(
        self,
        vix: Optional[float],
        adx: float,
        ewma_vol: float,
        drawdown: float,
    ) -> MarketRegime:
        """Assess current market regime and compute position scale."""
        # VIX checks
        if vix is None:
            vix = 20.0  # assume moderate if unavailable
        vix_ok = vix < self.vix_entry_max
        vix_danger = vix >= self.vix_force_close

        # ADX check
        is_trending = adx > self.adx_trend_threshold

        # Position scale from vol target
        if ewma_vol > 0:
            vol_scale = min(self.vol_target / ewma_vol, 1.0)
        else:
            vol_scale = 1.0

        # VIX continuous scaling (validated: reduces MaxDD from 60% to 38%)
        if vix >= self.vix_force_close:
            vix_scale = 0.0
        elif vix >= self.vix_entry_max:
            vix_scale = 0.25
        elif vix >= 20.0:
            vix_scale = 0.50
        elif vix >= 15.0:
            vix_scale = 0.75
        else:
            vix_scale = 1.0

        # Drawdown governor
        dd_scale = self.dd_scale_factor if drawdown < self.dd_threshold else 1.0

        position_scale = min(
            vol_scale * vix_scale * dd_scale,
            self.max_position_scale
        )

        # Regime label
        if vix_danger:
            label = "danger"
        elif not vix_ok:
            label = "high_vol"
        elif is_trending:
            label = "bull_trend"
        else:
            label = "bull_range"

        return MarketRegime(
            vix_level=vix,
            adx_value=adx,
            is_trending=is_trending,
            vix_ok=vix_ok,
            vix_danger=vix_danger,
            position_scale=round(position_scale, 3),
            regime_label=label,
        )

    def should_allow_entry(self, regime: MarketRegime, strategy_type: str = "trend") -> bool:
        """Determine if a new entry should be allowed given current regime.
        
        strategy_type: "trend" (momentum/breakout/ema_cross) or "reversion" (mean_reversion/rsi_reversal)
        """
        if regime.vix_danger:
            return False
        if not regime.vix_ok:
            return False
        if strategy_type == "trend" and not regime.is_trending:
            return False
        return True

    def adjust_position_size(self, base_allocation: float, regime: MarketRegime) -> float:
        """Scale position allocation by regime-derived factor."""
        return round(base_allocation * regime.position_scale, 4)
