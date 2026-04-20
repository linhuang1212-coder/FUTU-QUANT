import asyncio
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("telegram")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._bot = None

    async def _ensure_bot(self):
        if self._bot is None:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
            except ImportError:
                logger.warning("python-telegram-bot not installed, notifications disabled")
                self.enabled = False

    async def send_message(self, text: str) -> bool:
        if not self.enabled:
            logger.info(f"[Telegram disabled] {text}")
            return False
        try:
            await self._ensure_bot()
            if self._bot:
                await self._bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
                return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
        return False

    def send_sync(self, text: str) -> bool:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.send_message(text))

    def notify_system_start(self, balance: float) -> None:
        self.send_sync(f"*FUTU-QUANT started*\nBalance: ${balance:,.2f}")

    def notify_open_position(self, symbol: str, quantity: int, price: float, strategy: str, strength: float) -> None:
        self.send_sync(
            f"*OPEN*\n"
            f"Symbol: `{symbol}`\n"
            f"Qty: {quantity}\n"
            f"Price: ${price:.2f}\n"
            f"Strategy: {strategy}\n"
            f"Strength: {strength:.0f}"
        )

    def notify_close_position(self, symbol: str, quantity: int, price: float, pnl: float, pnl_pct: float) -> None:
        emoji = "+" if pnl >= 0 else ""
        self.send_sync(
            f"*CLOSE*\n"
            f"Symbol: `{symbol}`\n"
            f"Qty: {quantity}\n"
            f"Price: ${price:.2f}\n"
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)"
        )

    def notify_stop_loss(self, symbol: str, loss: float) -> None:
        self.send_sync(f"*STOP LOSS*\nSymbol: `{symbol}`\nLoss: ${loss:.2f}")

    def notify_pdt_warning(self, remaining: int) -> None:
        self.send_sync(f"*PDT WARNING*\nDay trades remaining: {remaining}")

    def notify_daily_summary(self, trades: int, pnl: float, balance: float) -> None:
        self.send_sync(
            f"*DAILY SUMMARY*\n"
            f"Trades: {trades}\n"
            f"PnL: ${pnl:+.2f}\n"
            f"Balance: ${balance:,.2f}"
        )

    def notify_error(self, error_msg: str) -> None:
        self.send_sync(f"*ERROR*\n{error_msg}")
