from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from data.downloader import (
    load_daily, save_daily, download_daily_yf, _normalize_symbol, DATA_DIR,
)
from utils.logger import setup_logger

logger = setup_logger("factor.data_provider")

CACHE_DIR = DATA_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class FactorDataProvider:
    """Unified data provider for factor analysis with local-first caching."""

    def __init__(self, quote_ctx=None, fmp_api_key: str = ""):
        self._quote_ctx = quote_ctx
        self._fmp_key = fmp_api_key

    def get_daily(self, symbol: str, years: int = 5) -> Optional[pd.DataFrame]:
        """Get daily OHLCV with automatic cache refresh.

        Priority: local CSV -> Yahoo Finance -> Futu API.
        Returns DataFrame with DatetimeIndex.
        """
        norm = _normalize_symbol(symbol)

        df = load_daily(norm)
        if df is not None and len(df) > 0:
            last_date = pd.to_datetime(df["time_key"].iloc[-1])
            if (datetime.now() - last_date).days <= 3:
                return self._to_indexed(df)

        try:
            raw = download_daily_yf(norm, years)
            if raw is not None and not raw.empty:
                csv_path = DATA_DIR / f"{norm}_daily.csv"
                raw.to_csv(csv_path, index=False)
                logger.info(f"Downloaded {len(raw)} bars for {norm} via Yahoo")
                return self._to_indexed(raw)
        except Exception as e:
            logger.debug(f"Yahoo download failed for {norm}: {e}")

        if self._quote_ctx:
            try:
                from futu import RET_OK, KLType
                time.sleep(0.5)
                ret, data, _ = self._quote_ctx.request_history_kline(
                    symbol, ktype=KLType.K_DAY, max_count=years * 252,
                )
                if ret == RET_OK and data is not None and len(data) > 50:
                    data = data.rename(columns={"time_key": "time_key"})
                    csv_path = DATA_DIR / f"{norm}_daily.csv"
                    data[["time_key", "open", "high", "low", "close", "volume"]].to_csv(
                        csv_path, index=False)
                    logger.info(f"Downloaded {len(data)} bars for {norm} via Futu")
                    return self._to_indexed(data)
            except Exception as e:
                logger.debug(f"Futu download failed for {symbol}: {e}")

        if df is not None and len(df) > 0:
            return self._to_indexed(df)
        return None

    def get_daily_panel(self, symbols: list[str],
                        years: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get aligned price and volume panels for multiple symbols."""
        price_dict = {}
        vol_dict = {}
        for sym in symbols:
            df = self.get_daily(sym, years)
            if df is None or df.empty:
                logger.warning(f"No data for {sym}, skipping")
                continue
            norm = _normalize_symbol(sym)
            price_dict[norm] = df["close"]
            vol_dict[norm] = df["volume"]

        if not price_dict:
            return pd.DataFrame(), pd.DataFrame()

        prices = pd.DataFrame(price_dict).sort_index().ffill()
        volumes = pd.DataFrame(vol_dict).sort_index().ffill()
        return prices, volumes

    def get_fundamentals(self, symbol: str) -> dict:
        """Get fundamental data from FMP API with 7-day cache."""
        norm = _normalize_symbol(symbol)
        cache_path = CACHE_DIR / f"fundamentals_{norm}.json"

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
                if (datetime.now() - cached_at).days < 7:
                    return data
            except Exception:
                pass

        if not self._fmp_key:
            return {}

        try:
            import requests
            url = f"https://financialmodelingprep.com/api/v3/profile/{norm}"
            resp = requests.get(url, params={"apikey": self._fmp_key}, timeout=10)
            if resp.status_code != 200 or not resp.json():
                return {}

            profile = resp.json()[0]
            result = {
                "pe": float(profile.get("pe", 0) or 0),
                "pb": float(profile.get("price", 0) or 0) / max(
                    float(profile.get("bookValuePerShare", 0) or 1), 0.01),
                "roe": 0.0,
                "revenue_growth": float(profile.get("revenueGrowth", 0) or 0),
                "market_cap": float(profile.get("mktCap", 0) or 0),
                "sector": profile.get("sector", ""),
                "_cached_at": datetime.now().isoformat(),
            }
            cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger.info(f"Cached fundamentals for {norm}")
            return result
        except Exception as e:
            logger.debug(f"FMP fetch failed for {norm}: {e}")
            return {}

    def get_fundamentals_panel(self, symbols: list[str]) -> pd.DataFrame:
        """Get fundamentals for multiple symbols as a DataFrame."""
        rows = {}
        for sym in symbols:
            data = self.get_fundamentals(sym)
            if data:
                norm = _normalize_symbol(sym)
                rows[norm] = {k: v for k, v in data.items() if not k.startswith("_")}
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).T

    @staticmethod
    def _to_indexed(df: pd.DataFrame) -> pd.DataFrame:
        """Convert time_key column to DatetimeIndex."""
        result = df.copy()
        result.index = pd.to_datetime(result["time_key"])
        result = result.sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
        return result
