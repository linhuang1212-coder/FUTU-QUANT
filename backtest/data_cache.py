from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional
from data.downloader import load_daily, save_daily


try:
    from data.indicators import TechnicalIndicators
    _HAS_INDICATORS = True
except Exception:
    _HAS_INDICATORS = False


class DataCache:
    """Singleton-style data cache: load once, reuse everywhere."""

    _instance: Optional[DataCache] = None

    def __init__(self):
        self._frames: dict[str, pd.DataFrame] = {}
        self._closes: dict[str, np.ndarray] = {}
        self._ivr: dict[str, np.ndarray] = {}
        self._rets: dict[str, np.ndarray] = {}
        self._indicators: dict[str, pd.DataFrame] = {}

    @classmethod
    def get(cls) -> DataCache:
        if cls._instance is None:
            cls._instance = DataCache()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def load_symbols(self, symbols: list[str], download_missing: bool = True, years: int = 10):
        """Load all symbols into cache. Downloads missing data if requested."""
        for sym in symbols:
            clean = sym.replace("US.", "")
            if clean in self._frames:
                continue
            df = load_daily(clean)
            if df is None or df.empty:
                if download_missing:
                    save_daily(clean, years=years)
                    df = load_daily(clean)
            if df is not None and not df.empty:
                self._frames[clean] = df
                self._closes[clean] = df["close"].values.astype(np.float64)
                self._rets[clean] = np.diff(np.log(self._closes[clean]))

    def get_closes(self, symbol: str) -> np.ndarray:
        clean = symbol.replace("US.", "")
        return self._closes.get(clean, np.array([]))

    def get_frame(self, symbol: str) -> pd.DataFrame:
        clean = symbol.replace("US.", "")
        return self._frames.get(clean, pd.DataFrame())

    def get_ivr(self, symbol: str) -> np.ndarray:
        clean = symbol.replace("US.", "")
        if clean not in self._ivr:
            closes = self._closes.get(clean)
            if closes is not None and len(closes) > 252:
                from backtest.turbo_core import precompute_ivr_fast
                self._ivr[clean] = precompute_ivr_fast(closes)
            else:
                self._ivr[clean] = np.array([])
        return self._ivr[clean]

    def get_indicators(self, symbol: str) -> pd.DataFrame:
        """Get DataFrame with all technical indicators computed."""
        clean = symbol.replace("US.", "")
        if clean not in self._indicators:
            df = self._frames.get(clean)
            if df is not None and not df.empty and _HAS_INDICATORS:
                try:
                    enriched = TechnicalIndicators.add_all(df.copy())
                    self._indicators[clean] = enriched
                except Exception:
                    self._indicators[clean] = pd.DataFrame()
            else:
                self._indicators[clean] = pd.DataFrame()
        return self._indicators[clean]

    def symbols(self) -> list[str]:
        return list(self._frames.keys())

    def summary(self) -> str:
        lines = [f"DataCache: {len(self._frames)} symbols loaded"]
        for sym in sorted(self._frames.keys()):
            n = len(self._closes[sym])
            ivr_status = "cached" if sym in self._ivr else "not computed"
            lines.append(f"  {sym}: {n} bars, IVR: {ivr_status}")
        return "\n".join(lines)
