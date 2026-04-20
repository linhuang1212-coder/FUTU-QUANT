from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class VwapReversionStrategy(BaseStrategy):
    """VWAP-anchored mean reversion for volatile / leveraged ETFs."""

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "dev_threshold": 2.0,
            "rsi_period": 10,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_period": 14,
            "use_volume_filter": True,
            "min_volume_ratio": 1.0,
        }
        if params:
            default_params.update(params)
        super().__init__("vwap_reversion", default_params)

    def _volume_ok(self, df: pd.DataFrame) -> bool:
        if not self.params["use_volume_filter"]:
            return True
        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        if not np.isfinite(vol_avg) or vol_avg <= 0:
            return False
        ratio = df["volume"].iloc[-1] / vol_avg
        return ratio >= self.params["min_volume_ratio"]

    @staticmethod
    def _finite(x: float) -> bool:
        return bool(np.isfinite(x))

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        rsi_p = int(self.params["rsi_period"])
        atr_p = int(self.params["atr_period"])
        min_len = max(20, atr_p, rsi_p) + 6
        if len(bar_data) < min_len:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_vwap(df)
        df = TechnicalIndicators.add_rsi(df, rsi_p)
        df = TechnicalIndicators.add_atr(df, atr_p)

        rsi_col = f"rsi_{rsi_p}"
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if not self._volume_ok(df):
            return None

        vwap = float(curr["vwap"])
        close = float(curr["close"])
        prev_close = float(prev["close"])
        prev_vwap = float(prev["vwap"])
        rsi_curr = float(curr[rsi_col])
        rsi_prev = float(prev[rsi_col])

        if not all(
            self._finite(x)
            for x in (vwap, close, prev_close, prev_vwap, rsi_curr, rsi_prev)
        ):
            return None
        if vwap == 0:
            return None

        dev = float(self.params["dev_threshold"])
        rsi_os = float(self.params["rsi_oversold"])
        rsi_ob = float(self.params["rsi_overbought"])
        lower = vwap * (1.0 - dev / 100.0)
        upper = vwap * (1.0 + dev / 100.0)
        vwap_dist_pct = (close - vwap) / vwap * 100.0

        recent = df.iloc[-6:-1]
        rsi_recent = recent[rsi_col].astype(float)
        vwap_recent = recent["vwap"].astype(float)
        close_recent = recent["close"].astype(float)
        if not (np.isfinite(rsi_recent).all() and np.isfinite(vwap_recent).all() and np.isfinite(close_recent).all()):
            return None

        recent_oversold = (
            (close_recent < vwap_recent * (1.0 - dev / 100.0)) & (rsi_recent < rsi_os)
        ).any()

        crossed_up_to_vwap = prev_close < prev_vwap and close >= vwap
        if recent_oversold and crossed_up_to_vwap:
            strength = float(min(85.0, 55.0 + min(abs(rsi_curr - rsi_os), 25.0) * 1.0))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"VWAP mean exit after stretch below (RSI {rsi_prev:.1f}→{rsi_curr:.1f})",
            )

        if close > upper and rsi_curr > rsi_ob:
            excess = max(0.0, vwap_dist_pct - dev)
            strength = float(min(95.0, 52.0 + excess * 3.5 + max(0.0, rsi_curr - rsi_ob) * 0.35))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Above VWAP+{dev:.1f}% & RSI overbought {rsi_curr:.1f} (dist {vwap_dist_pct:+.2f}%)",
            )

        if close < lower and rsi_curr < rsi_os:
            excess = max(0.0, -vwap_dist_pct - dev)
            strength = float(min(95.0, 52.0 + excess * 3.5 + max(0.0, rsi_os - rsi_curr) * 0.35))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"Below VWAP-{dev:.1f}% & RSI oversold {rsi_curr:.1f} (dist {vwap_dist_pct:+.2f}%)",
                suggested_type=SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK,
            )

        return None
