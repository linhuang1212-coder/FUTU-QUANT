"""Signal quality filter to reduce false entries.

Applies multiple confirmation checks to raw strategy signals:
1. Minimum strength threshold
2. Multi-strategy agreement (N strategies must agree)
3. Volume confirmation (above 20-day average)
4. ADX minimum for trend strategies
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FilterResult:
    passed: bool
    reason: str
    adjusted_strength: float


TREND_STRATEGIES = {"momentum", "breakout"}


class SignalFilter:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.min_strength = config.get("min_strength", 60.0)
        self.min_strategies_agree = config.get("min_strategies_agree", 2)
        self.adx_entry_min = config.get("adx_entry_min", 20.0)
        self.volume_confirm = config.get("volume_confirm", True)

    def filter_signals(
        self,
        signals: list[dict],
        adx: float = 0.0,
        df_map: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[dict]:
        """Filter a batch of signals for one evaluation cycle."""
        if not self.enabled:
            return signals

        # SELL signals pass through unfiltered to preserve exit timing
        sell_signals = [s for s in signals if s["signal"].direction.value == "SELL"]
        buy_signals = [s for s in signals if s["signal"].direction.value == "BUY"]

        groups: dict[str, list[dict]] = defaultdict(list)
        for s in buy_signals:
            groups[s["symbol"]].append(s)

        filtered_buys = []
        for sym, group in groups.items():
            strong_signals = [
                s for s in group if s["signal"].strength >= self.min_strength
            ]
            if not strong_signals:
                continue

            unique_strats = set(s["strategy_name"] for s in strong_signals)

            if len(unique_strats) < self.min_strategies_agree:
                continue

            trend_only = all(
                s["strategy_name"] in TREND_STRATEGIES for s in strong_signals
            )
            if trend_only and adx < self.adx_entry_min:
                continue

            if self.volume_confirm and df_map and sym in df_map:
                df = df_map[sym]
                if len(df) >= 20:
                    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
                    cur_vol = df["volume"].iloc[-1]
                    if cur_vol < vol_avg * 0.8:
                        continue

            best = max(strong_signals, key=lambda s: s["score"])
            agreement_bonus = (len(unique_strats) - 1) * 5.0
            best["signal"].strength = min(
                best["signal"].strength + agreement_bonus, 100.0
            )
            best["score"] = best["sharpe_weight"] * best["signal"].strength
            filtered_buys.append(best)

        return filtered_buys + sell_signals

    def filter_signals_vectorized(
        self,
        buy_scores: dict[str, np.ndarray],
        sell_scores: dict[str, np.ndarray],
        all_ind: dict[str, pd.DataFrame],
        symbols: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Vectorized version for backtest: only filter BUY signals.
        SELL signals pass through unmodified to preserve exit timing."""
        if not self.enabled:
            return buy_scores, sell_scores

        new_buy = {}
        for sym in symbols:
            bs = buy_scores[sym].copy()

            bs[bs < self.min_strength * 0.8] = 0

            if self.volume_confirm and sym in all_ind:
                df = all_ind[sym]
                vol_avg = df["volume"].rolling(20).mean().values
                cur_vol = df["volume"].values
                low_vol = cur_vol < vol_avg * 0.8
                low_vol = low_vol | np.isnan(vol_avg)
                bs[low_vol] = 0

            new_buy[sym] = bs

        return new_buy, sell_scores
