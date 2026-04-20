from typing import Optional

import pandas as pd

from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class MultiFactorStrategy(BaseStrategy):
    """Multi-indicator voting: BUY/SELL when enough factors align on direction."""

    # MACD warmup: slow span + signal span is a practical minimum for stable hist.
    _MACD_WARMUP_BARS = 35
    _VOLUME_AVG_WINDOW = 20

    def __init__(self, params: dict | None = None):
        default_params = {
            "fast_ma_period": 8,
            "slow_ma_period": 20,
            "rsi_period": 14,
            "ema_period": 20,
            "buy_threshold": 3,
            "sell_threshold": 3,
        }
        if params:
            default_params.update(params)
        super().__init__("multi_factor", default_params)

    def _min_bars_required(self) -> int:
        p = self.params
        return (
            max(
                p["slow_ma_period"],
                p["fast_ma_period"],
                p["rsi_period"],
                p["ema_period"],
                self._VOLUME_AVG_WINDOW,
                self._MACD_WARMUP_BARS,
            )
            + 5
        )

    @staticmethod
    def _strength_from_votes(vote_sum: int) -> float:
        return float(min(50 + abs(vote_sum) * 10, 95))

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        if len(bar_data) < self._min_bars_required():
            return None

        p = self.params
        df = bar_data.copy()

        df = TechnicalIndicators.add_ma(df, p["fast_ma_period"])
        df = TechnicalIndicators.add_ma(df, p["slow_ma_period"])
        df = TechnicalIndicators.add_rsi(df, p["rsi_period"])
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_ema(df, p["ema_period"])

        fast_col = f"ma_{p['fast_ma_period']}"
        slow_col = f"ma_{p['slow_ma_period']}"
        rsi_col = f"rsi_{p['rsi_period']}"
        ema_col = f"ema_{p['ema_period']}"

        row = df.iloc[-1]
        vol_avg = df["volume"].rolling(self._VOLUME_AVG_WINDOW).mean().iloc[-1]

        required = (
            row[fast_col],
            row[slow_col],
            row[rsi_col],
            row["macd_hist"],
            row[ema_col],
            row["close"],
            row["volume"],
            vol_avg,
        )
        if any(pd.isna(x) for x in required):
            return None

        vote_sum = 0
        parts: list[str] = []

        if row[fast_col] > row[slow_col]:
            vote_sum += 1
            parts.append("MA+")
        elif row[fast_col] < row[slow_col]:
            vote_sum -= 1
            parts.append("MA-")

        if row[rsi_col] > 50:
            vote_sum += 1
            parts.append("RSI+")
        elif row[rsi_col] < 50:
            vote_sum -= 1
            parts.append("RSI-")

        if row["macd_hist"] > 0:
            vote_sum += 1
            parts.append("MACD+")
        elif row["macd_hist"] < 0:
            vote_sum -= 1
            parts.append("MACD-")

        if row["close"] > row[ema_col]:
            vote_sum += 1
            parts.append("PxEMA+")
        elif row["close"] < row[ema_col]:
            vote_sum -= 1
            parts.append("PxEMA-")

        if row["volume"] > vol_avg:
            vote_sum += 1
            parts.append("Vol+")

        buy_th = p["buy_threshold"]
        sell_th = p["sell_threshold"]

        if vote_sum >= buy_th:
            strength = self._strength_from_votes(vote_sum)
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"votes={vote_sum} ({','.join(parts)}) ≥ buy {buy_th}",
                suggested_type=asset_type,
            )

        if vote_sum <= -sell_th:
            strength = self._strength_from_votes(vote_sum)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"votes={vote_sum} ({','.join(parts)}) ≤ -sell {sell_th}",
                suggested_type=SignalAssetType.STOCK,
            )

        return None
