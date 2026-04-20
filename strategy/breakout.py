from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            "lookback_period": 20,
            "volume_surge_ratio": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }
        if params:
            default_params.update(params)
        super().__init__("breakout", default_params)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        lookback = self.params["lookback_period"]
        if len(bar_data) < lookback + self.params["macd_slow"] + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_macd(df, self.params["macd_fast"], self.params["macd_slow"], self.params["macd_signal"])

        curr = df.iloc[-1]
        recent = df.iloc[-lookback - 1:-1]

        resistance = recent["high"].max()
        support = recent["low"].min()
        vol_avg = recent["volume"].mean()
        vol_ratio = curr["volume"] / vol_avg if vol_avg > 0 else 0

        if (curr["close"] > resistance and
            vol_ratio >= self.params["volume_surge_ratio"] and
            curr["macd_hist"] > 0):

            strength = min(50 + vol_ratio * 10 + curr["macd_hist"] * 5, 100)
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"Breakout above {resistance:.2f} + Volume {vol_ratio:.1f}x + MACD bullish",
                suggested_type=asset_type,
            )

        if (curr["close"] < support and
            vol_ratio >= self.params["volume_surge_ratio"] and
            curr["macd_hist"] < 0):

            strength = min(50 + vol_ratio * 10 + abs(curr["macd_hist"]) * 5, 100)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Breakdown below {support:.2f} + Volume {vol_ratio:.1f}x + MACD bearish",
            )

        return None
