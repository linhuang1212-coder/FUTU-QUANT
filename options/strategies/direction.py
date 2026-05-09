"""
Multi-signal direction analyzer for Credit Spread strategy.

Combines 4 signals to produce a directional score (-100 to +100):
  1. Trend (SMA 20/50 alignment) -- 40% weight
  2. Momentum (RSI 14) -- 25% weight
  3. MACD confirmation (histogram direction) -- 20% weight
  4. Volatility regime (ATR ratio) -- adjusts confidence

All thresholds are configurable for walk-forward optimization.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data.indicators import TechnicalIndicators
from utils.logger import setup_logger

logger = setup_logger("strategy.direction")

DEFAULT_PARAMS = {
    "trend_weight": 0.40,
    "momentum_weight": 0.25,
    "macd_weight": 0.20,
    "min_score": 25,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_bull_score": 50.0,
    "rsi_bear_score": -50.0,
    "rsi_overbought_score": -50.0,
    "rsi_oversold_score": 30.0,
    "trend_strong": 100.0,
    "trend_weak": 30.0,
    "atr_low_thresh": 0.8,
    "atr_high_thresh": 1.5,
    "atr_low_mult": 1.2,
    "atr_high_mult": 0.6,
}


@dataclass
class DirectionSignal:
    score: float          # -100 to +100
    direction: str        # "BULL", "BEAR", "NEUTRAL"
    trend_score: float
    momentum_score: float
    macd_score: float
    vol_multiplier: float
    details: str          # human-readable summary


class DirectionAnalyzer:
    """Analyze market direction using technical indicators.

    Works in two modes:
    1. Live mode: fetches data from Futu API (analyze method)
    2. Backtest mode: operates on a DataFrame (score_dataframe / score_row)
    """

    def __init__(self, quote_ctx=None, config: Optional[dict] = None):
        self._ctx = quote_ctx
        cfg = {**DEFAULT_PARAMS, **(config or {})}
        self.trend_weight = cfg["trend_weight"]
        self.momentum_weight = cfg["momentum_weight"]
        self.macd_weight = cfg["macd_weight"]
        self.min_score = cfg["min_score"]
        self.rsi_overbought = cfg["rsi_overbought"]
        self.rsi_oversold = cfg["rsi_oversold"]
        self.rsi_bull_score = cfg["rsi_bull_score"]
        self.rsi_bear_score = cfg["rsi_bear_score"]
        self.rsi_overbought_score = cfg["rsi_overbought_score"]
        self.rsi_oversold_score = cfg["rsi_oversold_score"]
        self.trend_strong = cfg["trend_strong"]
        self.trend_weak = cfg["trend_weak"]
        self.atr_low_thresh = cfg["atr_low_thresh"]
        self.atr_high_thresh = cfg["atr_high_thresh"]
        self.atr_low_mult = cfg["atr_low_mult"]
        self.atr_high_mult = cfg["atr_high_mult"]

    def get_params(self) -> dict:
        """Return current parameter dict (for serialization / logging)."""
        return {
            "trend_weight": self.trend_weight,
            "momentum_weight": self.momentum_weight,
            "macd_weight": self.macd_weight,
            "min_score": self.min_score,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
        }

    # ── Core scoring (single row) ──

    def _score_trend(self, price: float, sma20: float, sma50: float) -> tuple[float, str]:
        if price > sma20 and sma20 > sma50:
            return self.trend_strong, "强多头排列"
        elif price > sma20 and sma20 <= sma50:
            return self.trend_weak, "弱反弹(均线未金叉)"
        elif price < sma20 and sma20 < sma50:
            return -self.trend_strong, "强空头排列"
        elif price < sma20 and sma20 >= sma50:
            return -self.trend_weak, "弱回调(均线未死叉)"
        return 0.0, "中性"

    def _score_momentum(self, rsi: float) -> tuple[float, str]:
        if rsi > self.rsi_overbought:
            return self.rsi_overbought_score, f"超买({rsi:.0f})"
        elif rsi >= 50:
            return self.rsi_bull_score, f"偏多({rsi:.0f})"
        elif rsi >= self.rsi_oversold:
            return self.rsi_bear_score, f"偏空({rsi:.0f})"
        else:
            return self.rsi_oversold_score, f"超卖反弹({rsi:.0f})"

    def _score_macd(self, hist: float, hist_prev: float) -> tuple[float, str]:
        increasing = hist > hist_prev
        if hist > 0 and increasing:
            return 100.0, "多头动能增强"
        elif hist > 0 and not increasing:
            return 30.0, "多头动能减弱"
        elif hist < 0 and not increasing:
            return -100.0, "空头动能增强"
        elif hist < 0 and increasing:
            return -30.0, "空头动能减弱"
        return 0.0, "中性"

    def _vol_multiplier(self, atr: float, atr_avg: float) -> tuple[float, str]:
        ratio = atr / atr_avg if atr_avg > 0 else 1.0
        if ratio < self.atr_low_thresh:
            return self.atr_low_mult, f"低波动({ratio:.2f}x)信号可信度高"
        elif ratio > self.atr_high_thresh:
            return self.atr_high_mult, f"高波动({ratio:.2f}x)信号可信度低"
        return 1.0, f"正常波动({ratio:.2f}x)"

    def _compute_score(self, price, sma20, sma50, rsi, macd_hist, macd_hist_prev,
                       atr, atr_avg) -> DirectionSignal:
        trend, trend_label = self._score_trend(price, sma20, sma50)
        momentum, rsi_label = self._score_momentum(rsi)
        macd, macd_label = self._score_macd(macd_hist, macd_hist_prev)
        vol_mult, vol_label = self._vol_multiplier(atr, atr_avg)

        raw = trend * self.trend_weight + momentum * self.momentum_weight + macd * self.macd_weight
        final = max(-100.0, min(100.0, raw * vol_mult))

        if final > self.min_score:
            direction = "BULL"
        elif final < -self.min_score:
            direction = "BEAR"
        else:
            direction = "NEUTRAL"

        details = (
            f"趋势: {trend_label} ({trend:+.0f}×{self.trend_weight:.0%})\n"
            f"RSI: {rsi_label} ({momentum:+.0f}×{self.momentum_weight:.0%})\n"
            f"MACD: {macd_label} ({macd:+.0f}×{self.macd_weight:.0%})\n"
            f"波动率: {vol_label}\n"
            f"综合: {final:+.1f} → {direction}"
        )

        return DirectionSignal(
            score=round(final, 1), direction=direction,
            trend_score=trend, momentum_score=momentum,
            macd_score=macd, vol_multiplier=vol_mult, details=details,
        )

    # ── Live mode (Futu API) ──

    def analyze(self, symbol: str) -> Optional[DirectionSignal]:
        """Fetch daily klines from Futu and compute directional score."""
        df = self._fetch_daily_kline(symbol, count=80)
        if df is None or len(df) < 55:
            logger.warning(f"[DIRECTION] {symbol}: 日K数据不足，无法判断方向")
            return None

        df = self._add_indicators(df)
        signal = self._score_from_df(df, -1)

        logger.info(f"[DIRECTION] {symbol} score={signal.score:+.1f} -> {signal.direction} "
                    f"| trend={signal.trend_score:+.0f} rsi={signal.momentum_score:+.0f} "
                    f"macd={signal.macd_score:+.0f} vol={signal.vol_multiplier:.1f}x")
        return signal

    def analyze_with_factors(self, symbol: str) -> Optional[DirectionSignal]:
        """Factor-enhanced direction analysis.

        Uses IC-validated factors (MOM_3M, VOL_60D) alongside traditional
        technical indicators. Falls back to standard analyze on failure.
        """
        try:
            from factor.data_provider import FactorDataProvider
            from factor.technical import calc_momentum, calc_volatility

            provider = FactorDataProvider(quote_ctx=self._ctx)
            df_daily = provider.get_daily(symbol, years=1)
            if df_daily is None or len(df_daily) < 126:
                return self.analyze(symbol)

            closes = df_daily["close"]
            returns = closes.pct_change()

            mom_3m = closes.pct_change(63).iloc[-1]
            vol_60d = returns.rolling(60).std().iloc[-1] * np.sqrt(252)

            signal = self.analyze(symbol)
            if signal is None:
                return None

            factor_bias = 0.0
            if not pd.isna(mom_3m):
                factor_bias += 20.0 if mom_3m > 0.05 else (-20.0 if mom_3m < -0.05 else 0.0)
            if not pd.isna(vol_60d):
                factor_bias += -10.0 if vol_60d > 0.40 else (10.0 if vol_60d < 0.15 else 0.0)

            adjusted_score = max(-100, min(100, signal.score + factor_bias))
            if adjusted_score > self.min_score:
                new_dir = "BULL"
            elif adjusted_score < -self.min_score:
                new_dir = "BEAR"
            else:
                new_dir = "NEUTRAL"

            new_details = (f"{signal.details}\n"
                           f"因子增强: MOM_3M={mom_3m:+.1%} VOL_60D={vol_60d:.1%} "
                           f"bias={factor_bias:+.0f} -> {adjusted_score:+.1f}")

            logger.info(f"[DIRECTION+FACTOR] {symbol} "
                        f"base={signal.score:+.1f} factor_bias={factor_bias:+.1f} "
                        f"final={adjusted_score:+.1f} -> {new_dir}")

            return DirectionSignal(
                score=round(adjusted_score, 1),
                direction=new_dir,
                trend_score=signal.trend_score,
                momentum_score=signal.momentum_score,
                macd_score=signal.macd_score,
                vol_multiplier=signal.vol_multiplier,
                details=new_details,
            )
        except Exception as e:
            logger.debug(f"Factor-enhanced direction failed for {symbol}: {e}")
            return self.analyze(symbol)

    def _fetch_daily_kline(self, symbol: str, count: int = 80) -> Optional[pd.DataFrame]:
        from futu import RET_OK, KLType
        time.sleep(0.5)
        ret, data, _ = self._ctx.request_history_kline(
            symbol, ktype=KLType.K_DAY, max_count=count
        )
        if ret != RET_OK or data is None or data.empty:
            return None
        return data

    # ── Backtest mode (DataFrame) ──

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = TechnicalIndicators.add_ma(df, period=20)
        df = TechnicalIndicators.add_ma(df, period=50)
        df = TechnicalIndicators.add_rsi(df, period=14)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_atr(df, period=14)
        return df

    def _score_from_df(self, df: pd.DataFrame, idx: int) -> DirectionSignal:
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        atr_series = df["atr_14"].dropna()
        atr_avg = float(atr_series.iloc[max(0, len(atr_series) - 20):].mean())

        return self._compute_score(
            price=float(row["close"]),
            sma20=float(row["ma_20"]),
            sma50=float(row["ma_50"]),
            rsi=float(row["rsi_14"]),
            macd_hist=float(row["macd_hist"]),
            macd_hist_prev=float(prev["macd_hist"]),
            atr=float(row["atr_14"]),
            atr_avg=atr_avg,
        )

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every row of a daily OHLCV DataFrame (for backtesting).

        Returns a copy of df with added columns:
          dir_score, dir_direction, dir_trend, dir_momentum, dir_macd, dir_vol_mult
        Rows without enough lookback get NaN.
        """
        df = self._add_indicators(df.copy())

        scores = np.full(len(df), np.nan)
        directions = [""] * len(df)
        trends = np.full(len(df), np.nan)
        momentums = np.full(len(df), np.nan)
        macds = np.full(len(df), np.nan)
        vol_mults = np.full(len(df), np.nan)

        start_idx = 55  # need ~50 bars for SMA50 + a few for ATR warmup

        for i in range(start_idx, len(df)):
            atr_window = df["atr_14"].iloc[max(0, i - 19):i + 1].dropna()
            atr_avg = float(atr_window.mean()) if len(atr_window) > 0 else float(df["atr_14"].iloc[i])

            sig = self._compute_score(
                price=float(df["close"].iloc[i]),
                sma20=float(df["ma_20"].iloc[i]),
                sma50=float(df["ma_50"].iloc[i]),
                rsi=float(df["rsi_14"].iloc[i]),
                macd_hist=float(df["macd_hist"].iloc[i]),
                macd_hist_prev=float(df["macd_hist"].iloc[i - 1]),
                atr=float(df["atr_14"].iloc[i]),
                atr_avg=atr_avg,
            )
            scores[i] = sig.score
            directions[i] = sig.direction
            trends[i] = sig.trend_score
            momentums[i] = sig.momentum_score
            macds[i] = sig.macd_score
            vol_mults[i] = sig.vol_multiplier

        df["dir_score"] = scores
        df["dir_direction"] = directions
        df["dir_trend"] = trends
        df["dir_momentum"] = momentums
        df["dir_macd"] = macds
        df["dir_vol_mult"] = vol_mults
        return df
