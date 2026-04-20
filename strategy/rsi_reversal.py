from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class RsiReversalStrategy(BaseStrategy):
    """RSI mean-reversion style signals suited to volatile / leveraged ETFs."""

    def __init__(self, params: dict = None):
        default_params = {
            "rsi_period": 7,
            "rsi_buy_threshold": 30,
            "rsi_sell_threshold": 70,
            "rsi_exit_middle": 50,
            "use_volume_filter": True,
            "min_volume_ratio": 1.0,
            "atr_period": 14,
        }
        if params:
            default_params.update(params)
        super().__init__("rsi_reversal", default_params)

    def _volume_ok(self, df: pd.DataFrame) -> bool:
        if not self.params["use_volume_filter"]:
            return True
        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        if vol_avg <= 0:
            return False
        ratio = df["volume"].iloc[-1] / vol_avg
        return ratio >= self.params["min_volume_ratio"]

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        rsi_p = self.params["rsi_period"]
        atr_p = self.params["atr_period"]
        if len(bar_data) < max(20, atr_p) + rsi_p + 3:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_rsi(df, rsi_p)
        df = TechnicalIndicators.add_atr(df, atr_p)
        rsi_col = f"rsi_{rsi_p}"

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if not self._volume_ok(df):
            return None

        buy_th = self.params["rsi_buy_threshold"]
        sell_th = self.params["rsi_sell_threshold"]
        mid = self.params["rsi_exit_middle"]

        rsi_curr = float(curr[rsi_col])
        rsi_prev = float(prev[rsi_col])

        crossed_into_oversold = rsi_prev >= buy_th and rsi_curr < buy_th
        crossed_into_overbought = rsi_prev <= sell_th and rsi_curr > sell_th

        if crossed_into_oversold:
            extremity = max(0.0, buy_th - rsi_curr)
            strength = float(min(95.0, 50.0 + extremity * 2.2))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"RSI crossed below {buy_th} ({rsi_prev:.1f}→{rsi_curr:.1f})",
                suggested_type=SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK,
            )

        if crossed_into_overbought:
            extremity = max(0.0, rsi_curr - sell_th)
            strength = float(min(95.0, 50.0 + extremity * 2.2))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"RSI crossed above {sell_th} ({rsi_prev:.1f}→{rsi_curr:.1f})",
            )

        recent = df.iloc[-8:-1]
        was_oversold = (recent[rsi_col] < buy_th).any()
        exit_cross_up = rsi_prev < mid <= rsi_curr
        if was_oversold and exit_cross_up:
            strength = float(min(85.0, 52.0 + abs(rsi_curr - mid) * 1.5))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"RSI mean exit cross {mid} ({rsi_prev:.1f}→{rsi_curr:.1f})",
            )

        return None
