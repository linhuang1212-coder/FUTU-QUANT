from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            "lookback_period": 20,
            "volume_surge_ratio": 2.0,
            "volume_weak_ratio": 1.3,
            "volume_strong_ratio": 1.8,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "macd_required": False,
            "atr_breakout_enabled": True,
            "atr_breakout_multiplier": 1.5,
            "donchian_enabled": True,
        }
        if params:
            default_params.update(params)
        super().__init__("breakout", default_params)

    def _volume_strength(self, vol_ratio: float) -> tuple[float, str]:
        weak = self.params["volume_weak_ratio"]
        strong = self.params.get("volume_strong_ratio") or self.params.get(
            "volume_surge_ratio", 1.8
        )
        if vol_ratio < weak:
            return 0.0, "below weak"
        if vol_ratio < strong:
            return 55.0, "weak tier"
        return max(75.0, min(50 + vol_ratio * 12, 100)), "strong tier"

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        lookback = self.params["lookback_period"]
        atr_period = 14
        need = max(lookback + self.params["macd_slow"] + 5, lookback + atr_period + 5)
        if len(bar_data) < need:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_macd(
            df, self.params["macd_fast"], self.params["macd_slow"], self.params["macd_signal"]
        )
        df = TechnicalIndicators.add_atr(df, atr_period)

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        recent = df.iloc[-lookback - 1 : -1]

        resistance = recent["high"].max()
        support = recent["low"].min()
        donchian_high = recent["high"].max()
        donchian_low = recent["low"].min()

        vol_avg = recent["volume"].mean()
        vol_ratio = curr["volume"] / vol_avg if vol_avg > 0 else 0.0

        atr_col = f"atr_{atr_period}"
        atr_val = curr[atr_col] if pd.notna(curr[atr_col]) else 0.0
        atr_break_up = curr["close"] > prev["close"] + self.params["atr_breakout_multiplier"] * atr_val
        atr_break_down = curr["close"] < prev["close"] - self.params["atr_breakout_multiplier"] * atr_val

        resist_break = curr["close"] > resistance
        support_break = curr["close"] < support
        donch_up = self.params["donchian_enabled"] and curr["close"] > donchian_high
        donch_dn = self.params["donchian_enabled"] and curr["close"] < donchian_low

        price_bull = resist_break or (
            self.params["atr_breakout_enabled"] and atr_break_up
        ) or donch_up
        price_bear = support_break or (
            self.params["atr_breakout_enabled"] and atr_break_down
        ) or donch_dn

        base_strength, tier_note = self._volume_strength(vol_ratio)
        if base_strength <= 0:
            return None

        macd_ok_bull = curr["macd_hist"] > 0
        macd_ok_bear = curr["macd_hist"] < 0
        macd_bonus_bull = 10.0 if macd_ok_bull else 0.0
        macd_bonus_bear = 10.0 if macd_ok_bear else 0.0

        if price_bull:
            if self.params["macd_required"] and not macd_ok_bull:
                return None
            strength = base_strength + macd_bonus_bull
            strength = float(min(strength, 100))
            parts = []
            if resist_break:
                parts.append(f"resist>{resistance:.2f}")
            if self.params["atr_breakout_enabled"] and atr_break_up:
                parts.append("ATR impulse up")
            if donch_up:
                parts.append("Donchian high")
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"{'/'.join(parts)} | Vol {vol_ratio:.1f}x ({tier_note})"
                    f"{' +MACD' if macd_ok_bull else ''}"
                ),
                suggested_type=asset_type,
            )

        if price_bear:
            if self.params["macd_required"] and not macd_ok_bear:
                return None
            strength = base_strength + macd_bonus_bear
            strength = float(min(strength, 100))
            parts = []
            if support_break:
                parts.append(f"support<{support:.2f}")
            if self.params["atr_breakout_enabled"] and atr_break_down:
                parts.append("ATR impulse down")
            if donch_dn:
                parts.append("Donchian low")
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"{'/'.join(parts)} | Vol {vol_ratio:.1f}x ({tier_note})"
                    f"{' +MACD' if macd_ok_bear else ''}"
                ),
            )

        return None
