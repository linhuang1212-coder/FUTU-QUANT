import pandas as pd
from pathlib import Path
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("history")


class HistoryManager:
    def __init__(self, cache_dir: str = "data_store/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, ktype: str) -> Path:
        safe_symbol = symbol.replace(".", "_")
        return self.cache_dir / f"{safe_symbol}_{ktype}.csv"

    def save_to_cache(self, symbol: str, ktype: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol, ktype)
        df.to_csv(path, index=False)
        logger.info(f"Cached {len(df)} bars for {symbol} ({ktype})")

    def load_from_cache(self, symbol: str, ktype: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol, ktype)
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} bars from cache for {symbol} ({ktype})")
            return df
        return None

    def get_history(self, market_data, symbol: str, ktype: str = "K_DAY", count: int = 200, use_cache: bool = True) -> Optional[pd.DataFrame]:
        if use_cache:
            cached = self.load_from_cache(symbol, ktype)
            if cached is not None and len(cached) >= count:
                return cached.tail(count)

        data = market_data.get_kline(symbol, ktype, count)
        if data is not None:
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)
            self.save_to_cache(symbol, ktype, df)
            return df

        return self.load_from_cache(symbol, ktype)
