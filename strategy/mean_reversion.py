from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
        }
        if params:
            default_params.update(params)
        super().__init__("mean_reversion", default_params)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        if len(bar_data) < self.params["bb_period"] + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_bollinger(df, self.params["bb_period"], self.params["bb_std"])
        df = TechnicalIndicators.add_rsi(df, self.params["rsi_period"])

        rsi_col = f"rsi_{self.params['rsi_period']}"
        curr = df.iloc[-1]

        if (curr["close"] <= curr["bb_lower"] and
            curr[rsi_col] <= self.params["rsi_oversold"]):

            strength = min(50 + (self.params["rsi_oversold"] - curr[rsi_col]) * 2, 100)
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"Price below BB lower + RSI {curr[rsi_col]:.1f} oversold",
                suggested_type=asset_type,
            )

        if (curr["close"] >= curr["bb_upper"] and
            curr[rsi_col] >= self.params["rsi_overbought"]):

            strength = min(50 + (curr[rsi_col] - self.params["rsi_overbought"]) * 2, 100)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Price above BB upper + RSI {curr[rsi_col]:.1f} overbought",
            )

        return None
