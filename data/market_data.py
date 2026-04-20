from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("market_data")


class MarketData:
    def __init__(self, host: str = "127.0.0.1", port: int = 11111):
        self.host = host
        self.port = port
        self._quote_ctx = None
        self._connected = False

    def connect(self) -> bool:
        try:
            from futu import OpenQuoteContext
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
            self._connected = True
            logger.info(f"Connected to FutuOpenD at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to FutuOpenD: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._quote_ctx:
            self._quote_ctx.close()
            self._connected = False
            logger.info("Disconnected from FutuOpenD")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_snapshot(self, symbols: list[str]) -> Optional[dict]:
        if not self._connected or not self._quote_ctx:
            logger.error("Not connected to FutuOpenD")
            return None
        try:
            from futu import RET_OK
            ret, data = self._quote_ctx.get_market_snapshot(symbols)
            if ret == RET_OK:
                return data.to_dict("records")
            logger.error(f"get_snapshot failed: {data}")
        except Exception as e:
            logger.error(f"get_snapshot error: {e}")
        return None

    def get_kline(self, symbol: str, ktype: str = "K_1M", count: int = 100) -> Optional[dict]:
        if not self._connected or not self._quote_ctx:
            logger.error("Not connected to FutuOpenD")
            return None
        try:
            from futu import RET_OK, KLType
            kl_map = {
                "K_1M": KLType.K_1M,
                "K_5M": KLType.K_5M,
                "K_15M": KLType.K_15M,
                "K_60M": KLType.K_60M,
                "K_DAY": KLType.K_DAY,
            }
            kl = kl_map.get(ktype, KLType.K_1M)
            ret, data = self._quote_ctx.get_cur_kline(symbol, count, kl)
            if ret == RET_OK:
                return data
            logger.error(f"get_kline failed: {data}")
        except Exception as e:
            logger.error(f"get_kline error: {e}")
        return None

    def subscribe(self, symbols: list[str], sub_types: list[str] = None) -> bool:
        if not self._connected or not self._quote_ctx:
            return False
        try:
            from futu import RET_OK, SubType
            if sub_types is None:
                sub_types = [SubType.K_1M, SubType.QUOTE]
            ret, data = self._quote_ctx.subscribe(symbols, sub_types)
            if ret == RET_OK:
                logger.info(f"Subscribed to {symbols}")
                return True
            logger.error(f"subscribe failed: {data}")
        except Exception as e:
            logger.error(f"subscribe error: {e}")
        return False
