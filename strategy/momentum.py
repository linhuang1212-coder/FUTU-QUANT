from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class MomentumStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            "fast_ma_period": 5,
            "slow_ma_period": 20,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_ratio_threshold": 1.5,
        }
        if params:
            default_params.update(params)
        super().__init__("momentum", default_params)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        if len(bar_data) < self.params["slow_ma_period"] + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_ma(df, self.params["fast_ma_period"])
        df = TechnicalIndicators.add_ma(df, self.params["slow_ma_period"])
        df = TechnicalIndicators.add_rsi(df, self.params["rsi_period"])

        fast_col = f"ma_{self.params['fast_ma_period']}"
        slow_col = f"ma_{self.params['slow_ma_period']}"
        rsi_col = f"rsi_{self.params['rsi_period']}"

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = curr["volume"] / vol_avg if vol_avg > 0 else 0

        if (prev[fast_col] <= prev[slow_col] and
            curr[fast_col] > curr[slow_col] and
            curr[rsi_col] > self.params["rsi_oversold"] and
            vol_ratio >= self.params["volume_ratio_threshold"]):

            strength = min(50 + vol_ratio * 10 + (50 - abs(curr[rsi_col] - 50)) * 0.5, 100)
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"Golden cross + RSI {curr[rsi_col]:.1f} + Volume ratio {vol_ratio:.1f}x",
                suggested_type=asset_type,
            )

        if (prev[fast_col] >= prev[slow_col] and
            curr[fast_col] < curr[slow_col] and
            curr[rsi_col] < self.params["rsi_overbought"]):

            strength = min(50 + (curr[rsi_col] - 50) * 0.5, 100)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Death cross + RSI {curr[rsi_col]:.1f}",
            )

        return None
