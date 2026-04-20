from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class VolatilityBreakoutStrategy(BaseStrategy):
    """Bollinger squeeze followed by upper-band breakout with volume expansion."""

    def __init__(self, params: dict = None):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "squeeze_percentile": 20,
            "squeeze_lookback": 10,
            "vol_threshold": 1.2,
            "rsi_period": 14,
            "rsi_exit": 40,
            "width_lookback": 100,
        }
        if params:
            default_params.update(params)
        super().__init__("volatility_breakout", default_params)

    def _min_bars(self) -> int:
        p = self.params
        return max(
            p["bb_period"] + 5,
            p["width_lookback"] + p["bb_period"] + 5,
            p["rsi_period"] + 5,
            25,
        )

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        if len(bar_data) < self._min_bars():
            return None

        df = bar_data.copy()
        bb_p = self.params["bb_period"]
        bb_std = self.params["bb_std"]
        rsi_p = self.params["rsi_period"]
        width_lb = self.params["width_lookback"]
        sq_pct = self.params["squeeze_percentile"]
        sq_look = self.params["squeeze_lookback"]
        vol_thr = self.params["vol_threshold"]

        df = TechnicalIndicators.add_bollinger(df, period=bb_p, std=bb_std)
        df = TechnicalIndicators.add_rsi(df, period=rsi_p)

        mid = df["bb_middle"].replace(0, np.nan)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid

        q = float(sq_pct) / 100.0
        width_floor = df["bb_width"].rolling(window=width_lb, min_periods=width_lb).quantile(q)
        df["in_squeeze"] = (df["bb_width"] <= width_floor) & df["bb_width"].notna() & width_floor.notna()

        rsi_col = f"rsi_{rsi_p}"
        vol_avg_20 = df["volume"].rolling(20, min_periods=20).mean()

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if not pd.notna(curr[rsi_col]) or not pd.notna(vol_avg_20.iloc[-1]):
            return None

        vol_ok = vol_avg_20.iloc[-1] > 0 and curr["volume"] > vol_thr * vol_avg_20.iloc[-1]

        # Recent squeeze: any squeeze in the previous sq_look bars (exclude current bar)
        start = -(sq_look + 1)
        recent_slice = df.iloc[start:-1]
        had_recent_squeeze = recent_slice["in_squeeze"].any() if len(recent_slice) > 0 else False

        broke_above_upper = (
            curr["close"] > curr["bb_upper"]
            and prev["close"] <= prev["bb_upper"]
            and pd.notna(curr["bb_upper"])
            and pd.notna(prev["bb_upper"])
        )

        buy_ok = (
            had_recent_squeeze
            and broke_above_upper
            and vol_ok
            and curr[rsi_col] > 50
        )

        cross_below_lower = (
            curr["close"] < curr["bb_lower"]
            and prev["close"] >= prev["bb_lower"]
            and pd.notna(curr["bb_lower"])
            and pd.notna(prev["bb_lower"])
        )
        was_above_middle = prev["close"] > prev["bb_middle"] and pd.notna(prev["bb_middle"])

        sell_band = cross_below_lower and was_above_middle
        sell_rsi = curr[rsi_col] < self.params["rsi_exit"]

        if buy_ok:
            vol_ratio = curr["volume"] / vol_avg_20.iloc[-1] if vol_avg_20.iloc[-1] > 0 else 0.0
            strength = float(min(55.0 + (vol_ratio - vol_thr) * 15.0 + (curr[rsi_col] - 50) * 0.4, 100.0))
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"Squeeze→upper break | vol {vol_ratio:.2f}x>{vol_thr} | "
                    f"RSI {curr[rsi_col]:.1f}>50 | width ≤{sq_pct}%ile({width_lb})"
                ),
                suggested_type=asset_type,
            )

        if sell_band or sell_rsi:
            parts = []
            if sell_band:
                parts.append("close cross below bb_lower (from above middle)")
            if sell_rsi:
                parts.append(f"RSI {curr[rsi_col]:.1f}<{self.params['rsi_exit']}")
            strength = 60.0 if sell_band else float(min(50.0 + (self.params["rsi_exit"] - curr[rsi_col]) * 0.5, 90.0))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=" | ".join(parts),
            )

        return None
