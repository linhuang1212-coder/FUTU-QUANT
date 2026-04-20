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
            "cross_lookback": 3,
            "rsi_momentum_enabled": True,
            "ema_trend_enabled": True,
        }
        if params:
            default_params.update(params)
        super().__init__("momentum", default_params)

    @staticmethod
    def _had_ma_cross(
        df: pd.DataFrame, fast_col: str, slow_col: str, lookback: int, bullish: bool
    ) -> bool:
        for k in range(lookback):
            i_curr = -1 - k
            i_prev = -2 - k
            if abs(i_prev) > len(df):
                break
            p, c = df.iloc[i_prev], df.iloc[i_curr]
            if bullish:
                if p[fast_col] <= p[slow_col] and c[fast_col] > c[slow_col]:
                    return True
            else:
                if p[fast_col] >= p[slow_col] and c[fast_col] < c[slow_col]:
                    return True
        return False

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        lb = self.params["cross_lookback"]
        slow = self.params["slow_ma_period"]
        if len(bar_data) < slow + lb + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_ma(df, self.params["fast_ma_period"])
        df = TechnicalIndicators.add_ma(df, self.params["slow_ma_period"])
        df = TechnicalIndicators.add_ema(df, self.params["fast_ma_period"])
        df = TechnicalIndicators.add_ema(df, self.params["slow_ma_period"])
        df = TechnicalIndicators.add_rsi(df, self.params["rsi_period"])

        fast_col = f"ma_{self.params['fast_ma_period']}"
        slow_col = f"ma_{self.params['slow_ma_period']}"
        fast_ema = f"ema_{self.params['fast_ma_period']}"
        slow_ema = f"ema_{self.params['slow_ma_period']}"
        rsi_col = f"rsi_{self.params['rsi_period']}"

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        vol_avg_20 = df["volume"].rolling(20).mean().iloc[-1]
        vol_smooth = df["volume"].iloc[-3:].mean()
        vol_smooth_ratio = vol_smooth / vol_avg_20 if vol_avg_20 > 0 else 0.0

        candidates: list[Signal] = []

        had_golden = self._had_ma_cross(df, fast_col, slow_col, lb, bullish=True)
        if (
            had_golden
            and curr[rsi_col] > self.params["rsi_oversold"]
            and vol_smooth_ratio >= self.params["volume_ratio_threshold"]
        ):
            strength = min(
                50 + vol_smooth_ratio * 10 + (50 - abs(curr[rsi_col] - 50)) * 0.5, 100
            )
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            candidates.append(
                Signal(
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    strength=strength,
                    strategy_name=self.name,
                    reason=(
                        f"Golden cross (≤{lb} bars) + RSI {curr[rsi_col]:.1f} + "
                        f"Vol(3b)/SMA20 {vol_smooth_ratio:.1f}x"
                    ),
                    suggested_type=asset_type,
                )
            )

        if self.params["ema_trend_enabled"]:
            rsi_cross_50 = prev[rsi_col] < 50 <= curr[rsi_col]
            if curr[fast_ema] > curr[slow_ema] and rsi_cross_50:
                strength = min(
                    55 + (curr[rsi_col] - 50) * 0.8 + (curr[fast_ema] - curr[slow_ema])
                    / max(curr["close"], 1e-9)
                    * 500,
                    95,
                )
                asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
                candidates.append(
                    Signal(
                        symbol=symbol,
                        direction=SignalDirection.BUY,
                        strength=float(strength),
                        strategy_name=self.name,
                        reason=(
                            f"EMA uptrend + RSI cross 50 ({prev[rsi_col]:.1f}→{curr[rsi_col]:.1f})"
                        ),
                        suggested_type=asset_type,
                    )
                )

        if self.params["rsi_momentum_enabled"]:
            rsi_cross_50 = prev[rsi_col] < 50 <= curr[rsi_col]
            ema_already = (
                self.params["ema_trend_enabled"]
                and curr[fast_ema] > curr[slow_ema]
                and rsi_cross_50
            )
            if (
                rsi_cross_50
                and vol_smooth_ratio >= self.params["volume_ratio_threshold"]
                and not ema_already
            ):
                strength = float(
                    min(
                        max(55, 55 + (curr[rsi_col] - 50) * 0.6),
                        65,
                    )
                )
                candidates.append(
                    Signal(
                        symbol=symbol,
                        direction=SignalDirection.BUY,
                        strength=strength,
                        strategy_name=self.name,
                        reason=(
                            f"RSI momentum cross 50 + vol {vol_smooth_ratio:.1f}x (weak {strength:.0f})"
                        ),
                        suggested_type=SignalAssetType.STOCK,
                    )
                )

        if candidates:
            return max(candidates, key=lambda s: s.strength)

        had_death = self._had_ma_cross(df, fast_col, slow_col, lb, bullish=False)
        if had_death and curr[rsi_col] < self.params["rsi_overbought"]:
            strength = min(50 + (curr[rsi_col] - 50) * 0.5, 100)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Death cross (≤{lb} bars) + RSI {curr[rsi_col]:.1f}",
            )

        return None
