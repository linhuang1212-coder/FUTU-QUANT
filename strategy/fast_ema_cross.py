from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class FastEmaCrossStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        default_params = {
            "fast_ema_period": 3,
            "medium_ema_period": 8,
            "slow_ema_period": 21,
            "rsi_period": 10,
            "rsi_buy_floor": 40,
            "rsi_sell_ceiling": 60,
            "volume_ratio_threshold": 1.0,
        }
        if params:
            default_params.update(params)
        super().__init__("fast_ema_cross", default_params)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        p = self.params
        fast_p = int(p["fast_ema_period"])
        med_p = int(p["medium_ema_period"])
        slow_p = int(p["slow_ema_period"])
        rsi_p = int(p["rsi_period"])
        min_len = max(slow_p, rsi_p, 20) + 3
        if len(bar_data) < min_len:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_ema(df, fast_p)
        df = TechnicalIndicators.add_ema(df, med_p)
        df = TechnicalIndicators.add_ema(df, slow_p)
        df = TechnicalIndicators.add_rsi(df, rsi_p)

        fast_col = f"ema_{fast_p}"
        med_col = f"ema_{med_p}"
        slow_col = f"ema_{slow_p}"
        rsi_col = f"rsi_{rsi_p}"

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        for col in (fast_col, med_col, slow_col, rsi_col):
            if pd.isna(curr[col]) or pd.isna(prev[col]):
                return None

        vol_ma_20 = df["volume"].rolling(20).mean().iloc[-1]
        if vol_ma_20 <= 0 or pd.isna(vol_ma_20):
            return None
        vol_ratio = float(curr["volume"]) / float(vol_ma_20)

        fast_c, fast_prev = float(curr[fast_col]), float(prev[fast_col])
        med_c, med_prev = float(curr[med_col]), float(prev[med_col])
        slow_c = float(curr[slow_col])
        rsi_c = float(curr[rsi_col])

        cross_up = fast_prev <= med_prev and fast_c > med_c
        cross_down = fast_prev >= med_prev and fast_c < med_c

        v_th = float(p["volume_ratio_threshold"])
        if cross_up and med_c > slow_c and rsi_c > p["rsi_buy_floor"]:
            if v_th > 0 and vol_ratio < v_th:
                return None
            rsi_headroom = max(0.0, rsi_c - p["rsi_buy_floor"])
            base = 55.0 + min(20.0, rsi_headroom * 0.8)
            trend = max(0.0, (med_c - slow_c) / max(curr["close"], 1e-9)) * 400.0
            strength = float(min(95.0, base + min(12.0, trend)))
            if vol_ratio > 1.2:
                strength = float(min(100.0, strength + min(18.0, (vol_ratio - 1.2) * 35.0)))
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=(
                    f"Fast EMA cross up vs medium, medium>slow, RSI {rsi_c:.1f}, "
                    f"vol {vol_ratio:.2f}x 20d avg"
                ),
                suggested_type=SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK,
            )

        if cross_down and rsi_c < p["rsi_sell_ceiling"]:
            strength = float(
                min(
                    92.0,
                    52.0 + max(0.0, p["rsi_sell_ceiling"] - rsi_c) * 1.2,
                )
            )
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Fast EMA cross down vs medium, RSI {rsi_c:.1f}",
            )

        return None
