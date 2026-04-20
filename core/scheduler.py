from datetime import datetime
from zoneinfo import ZoneInfo
from utils.logger import setup_logger

logger = setup_logger("scheduler")


class TradingScheduler:
    def __init__(self, timezone: str = "US/Eastern", market_open: str = "09:30", market_close: str = "16:00"):
        self.tz = ZoneInfo(timezone)
        self.market_open = market_open
        self.market_close = market_close

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def is_market_hours(self) -> bool:
        now = self.now()
        if now.weekday() >= 5:
            return False
        time_str = now.strftime("%H:%M")
        return self.market_open <= time_str < self.market_close

    def minutes_to_close(self) -> int:
        now = self.now()
        close_h, close_m = map(int, self.market_close.split(":"))
        close_minutes = close_h * 60 + close_m
        now_minutes = now.hour * 60 + now.minute
        return close_minutes - now_minutes

    def should_force_close_day_trades(self, minutes_before: int = 15) -> bool:
        if not self.is_market_hours():
            return False
        return self.minutes_to_close() <= minutes_before
