from typing import Optional
import pandas as pd
import numpy as np
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
            "use_or_logic": True,
            "exit_at_middle": True,
        }
        if params:
            default_params.update(params)
        super().__init__("mean_reversion", default_params)

    @staticmethod
    def _bb_pct(row: pd.Series) -> float:
        upper, lower = row["bb_upper"], row["bb_lower"]
        band = upper - lower
        if not np.isfinite(band) or band <= 0:
            return 0.5
        return float((row["close"] - lower) / band)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        if len(bar_data) < self.params["bb_period"] + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_bollinger(df, self.params["bb_period"], self.params["bb_std"])
        df = TechnicalIndicators.add_rsi(df, self.params["rsi_period"])

        rsi_col = f"rsi_{self.params['rsi_period']}"
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        bb_pct = self._bb_pct(curr)

        use_or = self.params["use_or_logic"]

        bb_touch_lower = curr["close"] <= curr["bb_lower"]
        bb_touch_upper = curr["close"] >= curr["bb_upper"]
        rsi_os = curr[rsi_col] <= self.params["rsi_oversold"]
        rsi_ob = curr[rsi_col] >= self.params["rsi_overbought"]

        if use_or:
            buy_bb_only = bb_touch_lower and not rsi_os
            buy_rsi_only = rsi_os and not bb_touch_lower
            buy_both = bb_touch_lower and rsi_os
            if buy_both:
                strength = min(70 + (self.params["rsi_oversold"] - curr[rsi_col]) * 1.2, 90)
                asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    strength=float(strength),
                    strategy_name=self.name,
                    reason=(
                        f"BB lower + RSI oversold {curr[rsi_col]:.1f} (bb_pct={bb_pct:.2f})"
                    ),
                    suggested_type=asset_type,
                )
            if buy_bb_only:
                strength = float(min(max(50, 55 + (0.2 - min(bb_pct, 0.2)) * 50), 60))
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    strength=strength,
                    strategy_name=self.name,
                    reason=f"BB touch lower only (bb_pct={bb_pct:.2f})",
                    suggested_type=SignalAssetType.STOCK,
                )
            if buy_rsi_only:
                strength = float(
                    min(max(50, 55 + (self.params["rsi_oversold"] - curr[rsi_col]) * 0.35), 60)
                )
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    strength=strength,
                    strategy_name=self.name,
                    reason=f"RSI extreme oversold {curr[rsi_col]:.1f} (bb_pct={bb_pct:.2f})",
                    suggested_type=SignalAssetType.STOCK,
                )
        else:
            if bb_touch_lower and rsi_os:
                strength = min(50 + (self.params["rsi_oversold"] - curr[rsi_col]) * 2, 100)
                asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    strength=float(strength),
                    strategy_name=self.name,
                    reason=f"Price below BB lower + RSI {curr[rsi_col]:.1f} oversold",
                    suggested_type=asset_type,
                )

        if use_or:
            sell_bb_only = bb_touch_upper and not rsi_ob
            sell_rsi_only = rsi_ob and not bb_touch_upper
            sell_both = bb_touch_upper and rsi_ob
            if sell_both:
                strength = min(70 + (curr[rsi_col] - self.params["rsi_overbought"]) * 1.2, 90)
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    strength=float(strength),
                    strategy_name=self.name,
                    reason=(
                        f"BB upper + RSI overbought {curr[rsi_col]:.1f} (bb_pct={bb_pct:.2f})"
                    ),
                )
            if sell_bb_only:
                strength = float(min(max(50, 55 + (max(bb_pct, 0.8) - 0.8) * 50), 60))
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    strength=strength,
                    strategy_name=self.name,
                    reason=f"BB touch upper only (bb_pct={bb_pct:.2f})",
                )
            if sell_rsi_only:
                strength = float(
                    min(max(50, 55 + (curr[rsi_col] - self.params["rsi_overbought"]) * 0.35), 60)
                )
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    strength=strength,
                    strategy_name=self.name,
                    reason=f"RSI extreme overbought {curr[rsi_col]:.1f} (bb_pct={bb_pct:.2f})",
                )
        else:
            if bb_touch_upper and rsi_ob:
                strength = min(50 + (curr[rsi_col] - self.params["rsi_overbought"]) * 2, 100)
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    strength=float(strength),
                    strategy_name=self.name,
                    reason=f"Price above BB upper + RSI {curr[rsi_col]:.1f} overbought",
                )

        if self.params["exit_at_middle"]:
            recent = df.iloc[-6:-1]
            long_ctx = (recent["close"] <= recent["bb_lower"]).any() or (
                recent[rsi_col] <= self.params["rsi_oversold"] + 5
            ).any()
            short_ctx = (recent["close"] >= recent["bb_upper"]).any() or (
                recent[rsi_col] >= self.params["rsi_overbought"] - 5
            ).any()

            if (
                long_ctx
                and prev["close"] < prev["bb_middle"]
                and curr["close"] >= curr["bb_middle"]
            ):
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    strength=60.0,
                    strategy_name=self.name,
                    reason=f"Exit at BB middle after lower stretch (bb_pct={bb_pct:.2f})",
                )
            if (
                short_ctx
                and prev["close"] > prev["bb_middle"]
                and curr["close"] <= curr["bb_middle"]
            ):
                return Signal(
                    symbol=symbol,
                    direction=SignalDirection.SELL,
                    strength=60.0,
                    strategy_name=self.name,
                    reason=f"Exit at BB middle after upper stretch (bb_pct={bb_pct:.2f})",
                )

        return None
