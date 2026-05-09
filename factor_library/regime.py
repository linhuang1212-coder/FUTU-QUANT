"""
HMM Market Regime Detection — 隐马尔可夫模型市场状态识别

基于 SPY 的 log returns + 波动率 + RSI 训练 Gaussian HMM，
识别 5 种市场状态：
  0: CRISIS     — 高波动+大幅下跌
  1: BEAR       — 温和下跌+偏高波动
  2: NEUTRAL    — 窄幅震荡
  3: CALM_BULL  — 温和上涨+低波动
  4: STRONG_BULL — 强势上涨

参考: vigp17/market-regime-detection (GitHub, 2026)
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("regime")

MODEL_PATH = Path(__file__).resolve().parent.parent / "data_store" / "regime_hmm.pkl"

REGIME_NAMES = {
    "CRISIS": 0,
    "BEAR": 1,
    "NEUTRAL": 2,
    "CALM_BULL": 3,
    "STRONG_BULL": 4,
}
REGIME_ID_TO_NAME = {v: k for k, v in REGIME_NAMES.items()}

# Mapping to simplified 3-state for strategy decisions
REGIME_TO_STATE = {
    "CRISIS": "BEARISH",
    "BEAR": "BEARISH",
    "NEUTRAL": "NEUTRAL",
    "CALM_BULL": "BULLISH",
    "STRONG_BULL": "BULLISH",
}


def _compute_features(prices: pd.Series, window: int = 21) -> pd.DataFrame:
    """Compute HMM features from a price series.

    Features:
      - log_return: daily log return
      - volatility: rolling std of log returns (annualized)
      - rsi: 14-day RSI
      - sma_dist: distance from 50-day SMA (% terms)
    """
    log_ret = np.log(prices / prices.shift(1))

    vol = log_ret.rolling(window).std() * np.sqrt(252)

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    sma50 = prices.rolling(50).mean()
    sma_dist = (prices - sma50) / sma50

    features = pd.DataFrame({
        "log_return": log_ret,
        "volatility": vol,
        "rsi": rsi / 100.0,  # normalize to 0-1
        "sma_dist": sma_dist,
    }).dropna()

    return features


class RegimeDetector:
    """HMM-based market regime detector."""

    def __init__(self, n_regimes: int = 5, model_path: Path = MODEL_PATH):
        self.n_regimes = n_regimes
        self.model_path = model_path
        self._model = None
        self._regime_order: Optional[dict[int, str]] = None

    def train(self, prices: pd.Series, n_iter: int = 200) -> dict:
        """Train HMM on historical price data.

        Args:
            prices: Daily close prices (e.g. SPY)
            n_iter: EM iterations

        Returns:
            Training summary dict
        """
        from hmmlearn.hmm import GaussianHMM

        features = _compute_features(prices)
        if len(features) < 100:
            raise ValueError(f"Need >= 100 data points, got {len(features)}")

        X = features.values

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=False,
        )
        model.fit(X)

        states = model.predict(X)
        features["state"] = states

        # Order regimes by mean return: lowest = CRISIS, highest = STRONG_BULL
        mean_returns = features.groupby("state")["log_return"].mean().sort_values()
        order = {}
        names = ["CRISIS", "BEAR", "NEUTRAL", "CALM_BULL", "STRONG_BULL"]
        for i, (state_id, _) in enumerate(mean_returns.items()):
            if i < len(names):
                order[int(state_id)] = names[i]

        self._model = model
        self._regime_order = order

        self._save()

        # Compute stats per regime
        stats = {}
        for state_id, name in order.items():
            mask = features["state"] == state_id
            subset = features[mask]
            stats[name] = {
                "count": int(mask.sum()),
                "pct": round(mask.mean() * 100, 1),
                "avg_return": round(subset["log_return"].mean() * 252 * 100, 2),
                "avg_vol": round(subset["volatility"].mean() * 100, 1),
            }

        logger.info(f"[HMM] Trained on {len(features)} days, "
                    f"{self.n_regimes} regimes, score={model.score(X):.0f}")
        for name, s in stats.items():
            logger.info(f"  {name:12s}: {s['pct']:5.1f}% of days, "
                        f"ann.ret={s['avg_return']:+6.1f}%, "
                        f"vol={s['avg_vol']:.1f}%")

        return {
            "n_days": len(features),
            "score": model.score(X),
            "regimes": stats,
            "current": order.get(int(states[-1]), "NEUTRAL"),
        }

    def detect(self, prices: pd.Series) -> dict:
        """Detect current market regime from recent prices.

        Returns:
            dict with regime, state, confidence, history
        """
        self._ensure_loaded()
        if self._model is None:
            return self._fallback()

        features = _compute_features(prices)
        if len(features) < 10:
            return self._fallback()

        X = features.values
        states = self._model.predict(X)
        probs = self._model.predict_proba(X)

        current_state = int(states[-1])
        current_name = self._regime_order.get(current_state, "NEUTRAL")
        current_probs = probs[-1]

        # Recent regime distribution (last 20 days)
        recent = states[-20:]
        recent_dist = {}
        for sid, name in self._regime_order.items():
            recent_dist[name] = round((recent == sid).mean() * 100, 1)

        # Confidence = probability of the detected state
        confidence = float(current_probs[current_state])

        # Simplified 3-state for strategy decisions
        market_state = REGIME_TO_STATE.get(current_name, "NEUTRAL")

        result = {
            "regime": current_name,
            "market_state": market_state,
            "confidence": round(confidence, 3),
            "recent_distribution": recent_dist,
            "features": {
                "log_return": round(float(features["log_return"].iloc[-1]), 5),
                "volatility": round(float(features["volatility"].iloc[-1]), 4),
                "rsi": round(float(features["rsi"].iloc[-1]), 4),
                "sma_dist": round(float(features["sma_dist"].iloc[-1]), 4),
            },
        }

        logger.info(f"[HMM] Regime={current_name} ({market_state}) "
                    f"conf={confidence:.1%}")
        return result

    def _fallback(self) -> dict:
        return {
            "regime": "NEUTRAL",
            "market_state": "NEUTRAL",
            "confidence": 0.0,
            "recent_distribution": {},
            "features": {},
        }

    def _save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "regime_order": self._regime_order,
                "n_regimes": self.n_regimes,
                "trained_at": datetime.now().isoformat(),
            }, f)
        logger.info(f"[HMM] Model saved to {self.model_path}")

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self._model = data["model"]
                self._regime_order = data["regime_order"]
                self.n_regimes = data.get("n_regimes", 5)
                logger.info(f"[HMM] Model loaded (trained {data.get('trained_at', '?')})")
            except Exception as e:
                logger.warning(f"[HMM] Failed to load model: {e}")


def train_regime_model(years: int = 10) -> dict:
    """Convenience: download SPY data and train HMM.

    Called from CLI or scheduler.
    """
    import yfinance as yf

    logger.info(f"[HMM] Downloading SPY data ({years} years)...")
    spy = yf.download("SPY", period=f"{years}y", progress=False)
    if spy.empty or len(spy) < 500:
        raise ValueError(f"Insufficient SPY data: {len(spy)} rows")

    close = spy["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    detector = RegimeDetector()
    result = detector.train(close)
    return result


def get_current_regime() -> dict:
    """Convenience: detect current regime using recent SPY data."""
    import yfinance as yf

    spy = yf.download("SPY", period="1y", progress=False)
    if spy.empty:
        return {"regime": "NEUTRAL", "market_state": "NEUTRAL", "confidence": 0}

    close = spy["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    detector = RegimeDetector()
    return detector.detect(close)
