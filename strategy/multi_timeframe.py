"""Multi-timeframe strategy wrapper.

Allows a strategy to consult a higher-timeframe trend filter before
generating signals on the primary (lower) timeframe.
"""

from typing import Optional
import pandas as pd

from strategy.base import BaseStrategy, Signal, SignalDirection
from data.indicators import TechnicalIndicators


class MultiTimeframeStrategy(BaseStrategy):
    """Wraps an existing strategy and adds a higher-timeframe trend filter.

    The higher-timeframe data is resampled from the same source and checked
    for trend direction using a configurable EMA. Signals from the inner
    strategy are only passed through when they align with the HTF trend.
    """

    def __init__(
        self,
        inner_strategy: BaseStrategy,
        htf_rule: str = "1W",
        htf_ema_period: int = 10,
        require_alignment: bool = True,
        params: dict = None,
    ):
        merged = {
            "htf_rule": htf_rule,
            "htf_ema_period": htf_ema_period,
            "require_alignment": require_alignment,
            **inner_strategy.get_params(),
        }
        if params:
            merged.update(params)
        super().__init__(f"mtf_{inner_strategy.name}", merged)

        self.inner = inner_strategy
        self._htf_cache: Optional[pd.DataFrame] = None
        self._htf_cache_len: int = 0

    def _get_htf_trend(self, bar_data: pd.DataFrame) -> Optional[str]:
        """Return 'up', 'down', or None based on higher timeframe EMA slope."""
        if "time_key" not in bar_data.columns:
            return None

        need_refresh = (
            self._htf_cache is None
            or len(bar_data) != self._htf_cache_len
        )

        if need_refresh:
            htf = TechnicalIndicators.resample_kline(
                bar_data, self.params["htf_rule"]
            )
            if len(htf) < self.params["htf_ema_period"] + 1:
                return None
            htf = TechnicalIndicators.add_ema(htf, self.params["htf_ema_period"])
            self._htf_cache = htf
            self._htf_cache_len = len(bar_data)

        htf = self._htf_cache
        ema_col = f"ema_{self.params['htf_ema_period']}"
        if ema_col not in htf.columns or len(htf) < 2:
            return None

        curr_ema = htf[ema_col].iloc[-1]
        prev_ema = htf[ema_col].iloc[-2]

        if pd.isna(curr_ema) or pd.isna(prev_ema):
            return None
        if curr_ema > prev_ema:
            return "up"
        if curr_ema < prev_ema:
            return "down"
        return None

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        signal = self.inner.on_bar(symbol, bar_data)
        if signal is None:
            return None

        if not self.params.get("require_alignment", True):
            return signal

        trend = self._get_htf_trend(bar_data)
        if trend is None:
            return signal

        if signal.direction == SignalDirection.BUY and trend == "up":
            return signal
        if signal.direction == SignalDirection.SELL and trend == "down":
            return signal
        if signal.direction == SignalDirection.BUY and trend == "down":
            signal.strength = max(signal.strength * 0.6, 10)
            signal.reason += " [HTF contra-trend, reduced]"
            return signal
        if signal.direction == SignalDirection.SELL and trend == "up":
            signal.strength = max(signal.strength * 0.6, 10)
            signal.reason += " [HTF contra-trend, reduced]"
            return signal

        return signal
