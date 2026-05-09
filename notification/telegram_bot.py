import asyncio
import html
import re
import threading
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("telegram")


_ALLOWED_TAGS = re.compile(r"</?(?:b|i|u|s|code|pre|a)(?:\s[^>]*)?>", re.IGNORECASE)


def _sanitize_html(text: str) -> str:
    """Escape HTML special chars in user content while preserving allowed tags."""
    parts = _ALLOWED_TAGS.split(text)
    tags = _ALLOWED_TAGS.findall(text)
    result = []
    for i, part in enumerate(parts):
        result.append(html.escape(part))
        if i < len(tags):
            result.append(tags[i])
    return "".join(result)


class TelegramNotifier:
    def __init__(self, bot_token: str = "", chat_id: str = "", enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token and chat_id)
        self._bot = None

    async def _ensure_bot(self):
        if self._bot is None:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
            except ImportError:
                logger.warning("python-telegram-bot not installed, notifications disabled")
                self.enabled = False

    async def _send_parts(self, text: str, parse_mode: str | None) -> bool:
        if len(text) > 4000:
            parts = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for part in parts:
                await self._bot.send_message(
                    chat_id=self.chat_id, text=part, parse_mode=parse_mode)
        else:
            await self._bot.send_message(
                chat_id=self.chat_id, text=text, parse_mode=parse_mode)
        return True

    async def send_message(self, text: str) -> bool:
        if not self.enabled:
            logger.info(f"[Telegram disabled] {text}")
            return False
        try:
            await self._ensure_bot()
            if self._bot:
                sanitized = _sanitize_html(text)
                try:
                    return await self._send_parts(sanitized, parse_mode="HTML")
                except Exception:
                    plain = re.sub(r"<[^>]+>", "", text)
                    return await self._send_parts(plain, parse_mode=None)
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
        self.send_sync(f"<b>FUTU-QUANT 启动</b>\n余额: ${balance:,.2f}")

    def notify_open_position(self, symbol: str, quantity: int, price: float, strategy: str, strength: float) -> None:
        self.send_sync(
            f"<b>开仓</b>\n"
            f"标的: <code>{symbol}</code>\n"
            f"数量: {quantity}\n"
            f"价格: ${price:.2f}\n"
            f"策略: {strategy}\n"
            f"信号强度: {strength:.0f}"
        )

    def notify_close_position(self, symbol: str, quantity: int, price: float, pnl: float, pnl_pct: float) -> None:
        self.send_sync(
            f"<b>平仓</b>\n"
            f"标的: <code>{symbol}</code>\n"
            f"数量: {quantity}\n"
            f"价格: ${price:.2f}\n"
            f"盈亏: ${pnl:+.2f} ({pnl_pct:+.1f}%)"
        )

    def notify_stop_loss(self, symbol: str, loss: float) -> None:
        self.send_sync(f"<b>止损触发</b>\n标的: <code>{symbol}</code>\n亏损: ${loss:.2f}")

    def notify_pdt_warning(self, remaining: int) -> None:
        self.send_sync(f"<b>PDT 警告</b>\n剩余日内交易次数: {remaining}")

    def notify_daily_summary(self, trades: int, pnl: float, balance: float) -> None:
        self.send_sync(
            f"<b>每日汇总</b>\n"
            f"交易笔数: {trades}\n"
            f"盈亏: ${pnl:+.2f}\n"
            f"余额: ${balance:,.2f}"
        )

    def notify_daily_report(self, report_text: str) -> None:
        """Send a pre-formatted daily report via Telegram."""
        self.send_sync(report_text)

    def notify_error(self, error_msg: str) -> None:
        self.send_sync(f"<b>系统错误</b>\n{error_msg}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Interactive Bot — command handler with polling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TelegramCommandBot:
    """Interactive Telegram bot for factor library queries.

    Supported commands:
      /start      - Show available commands
      /screen     - Multi-factor stock screening (e.g. /screen momentum 10)
      /similar    - Find similar stocks (e.g. /similar AAPL)
      /report     - Generate daily factor report
      /anomalies  - Detect factor anomaly stocks
      /timing     - Market timing signal
      /status     - System status
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start_polling(self):
        """Start bot polling in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("[TelegramBot] 交互式命令轮询已启动")

    def stop_polling(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        """Long-polling loop for incoming messages."""
        import time
        try:
            from telegram import Bot
        except ImportError:
            logger.warning("[TelegramBot] python-telegram-bot not installed")
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot = Bot(token=self.bot_token)
        last_update_id = 0

        while self._running:
            try:
                updates = loop.run_until_complete(
                    bot.get_updates(offset=last_update_id + 1, timeout=10))
                for update in updates:
                    last_update_id = update.update_id
                    if update.message and update.message.text:
                        chat_id = str(update.message.chat_id)
                        if chat_id != self.chat_id:
                            continue
                        text = update.message.text.strip()
                        reply = self._handle_command(text)
                        if reply:
                            sanitized_reply = _sanitize_html(reply)
                            try:
                                loop.run_until_complete(
                                    bot.send_message(
                                        chat_id=self.chat_id, text=sanitized_reply,
                                        parse_mode="HTML"))
                            except Exception:
                                plain = re.sub(r"<[^>]+>", "", reply)
                                loop.run_until_complete(
                                    bot.send_message(
                                        chat_id=self.chat_id, text=plain,
                                        parse_mode=None))
            except Exception as e:
                logger.debug(f"[TelegramBot] Poll error: {e}")
                time.sleep(5)

    def _handle_command(self, text: str) -> str:
        """Route command text to handler and return reply."""
        parts = text.split()
        cmd = parts[0].lower() if parts else ""

        handlers = {
            "/start": self._cmd_start,
            "/help": self._cmd_start,
            "/screen": lambda: self._cmd_screen(parts[1:]),
            "/similar": lambda: self._cmd_similar(parts[1:]),
            "/report": self._cmd_report,
            "/anomalies": self._cmd_anomalies,
            "/timing": self._cmd_timing,
            "/status": self._cmd_status,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                return handler()
            except Exception as e:
                return f"<b>错误</b>\n{html.escape(str(e)[:500])}"
        elif cmd.startswith("/"):
            return self._cmd_start()
        return ""

    def _cmd_start(self) -> str:
        return (
            "<b>FUTU-QUANT 因子库 Bot</b>\n\n"
            "可用命令:\n"
            "/screen [model] [n] — 多因子选股\n"
            "  模型: value, momentum, quality,\n"
            "  low_risk, credit_spread\n"
            "  例: /screen momentum 10\n\n"
            "/similar [TICKER] — 找相似股票\n"
            "  例: /similar AAPL\n\n"
            "/report — 每日因子报告\n"
            "/anomalies — 因子异常股\n"
            "/timing — 市场择时信号\n"
            "/status — 系统状态"
        )

    def _load_matrix(self):
        from factor_library.storage import load_factors
        from factor_library.search import build_factor_matrix

        categories = ["technical", "risk", "volatility", "liquidity", "fundamental"]
        factor_dfs = {}
        for cat in categories:
            df = load_factors(cat)
            if not df.empty:
                factor_dfs[cat] = df
        if not factor_dfs:
            return None
        return build_factor_matrix(factor_dfs)

    def _cmd_screen(self, args: list[str]) -> str:
        model = args[0] if args else "momentum"
        top_n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 10

        from factor_library.screener import MODELS
        if model not in MODELS:
            return f"未知模型: {model}\n可选: {', '.join(MODELS.keys())}"

        matrix = self._load_matrix()
        if matrix is None:
            return "因子数据不可用, 请先运行因子计算"

        from factor_library.screener import score_stocks
        results = score_stocks(matrix, model=model, top_n=top_n)
        if results.empty:
            return "无结果"

        desc = MODELS[model]["description"]
        lines = [f"<b>{desc}</b>\n"]
        for _, row in results.iterrows():
            lines.append(
                f"{row['rank']:2.0f}. <code>{row['symbol']:6s}</code> "
                f"score={row['score']:.4f}")
        return "\n".join(lines)

    def _cmd_similar(self, args: list[str]) -> str:
        if not args:
            return "用法: /similar AAPL"
        symbol = args[0].upper()

        matrix = self._load_matrix()
        if matrix is None:
            return "因子数据不可用"

        from factor_library.search import find_similar
        results = find_similar(matrix, symbol, top_n=10)
        if results.empty:
            return f"{symbol} 不在因子库中"

        lines = [f"<b>与 {symbol} 最相似的股票</b>\n"]
        for i, (_, row) in enumerate(results.iterrows(), 1):
            lines.append(
                f"{i:2d}. <code>{row['symbol']:6s}</code> "
                f"similarity={row['similarity']:.4f}")
        return "\n".join(lines)

    def _cmd_report(self) -> str:
        matrix = self._load_matrix()
        if matrix is None:
            return "因子数据不可用"

        from factor_library.screener import generate_daily_report
        report = generate_daily_report(matrix)
        # Strip emoji for cleaner Telegram display, keep structure
        return f"<pre>{report[:3800]}</pre>"

    def _cmd_anomalies(self) -> str:
        matrix = self._load_matrix()
        if matrix is None:
            return "因子数据不可用"

        from factor_library.search import find_anomalies
        results = find_anomalies(matrix, top_n=10)
        if results.empty:
            return "无异常"

        lines = ["<b>因子异常 Top 10</b>\n"]
        for i, (_, row) in enumerate(results.iterrows(), 1):
            lines.append(
                f"{i:2d}. <code>{row['symbol']:6s}</code> "
                f"score={row['anomaly_score']:.2f}\n"
                f"    {row['extreme_factors']}")
        return "\n".join(lines)

    def _cmd_timing(self) -> str:
        matrix = self._load_matrix()
        if matrix is None:
            return "因子数据不可用"

        from factor_library.screener import market_timing_signal
        timing = market_timing_signal(matrix)

        state_emoji = {"BULLISH": "🟢", "NEUTRAL": "🟡", "BEARISH": "🔴"}
        emoji = state_emoji.get(timing["market_state"], "")

        lines = [f"<b>{emoji} 市场状态: {timing['market_state']}</b>",
                 f"综合评分: {timing['score']}"]
        for k, v in timing["signals"].items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append(f"\n{timing['recommendation']}")
        return "\n".join(lines)

    def _cmd_status(self) -> str:
        lines = ["<b>FUTU-QUANT 系统状态</b>\n"]

        try:
            from factor_library.storage import get_storage_stats
            stats = get_storage_stats()
            lines.append(f"因子库: {stats['total_mb']:.0f} MB")
            lines.append(f"  行情: {len(stats['parquet_files'])} 个文件")
            lines.append(f"  因子: {len(stats['factor_files'])} 个文件")
        except Exception:
            lines.append("因子库: 不可用")

        try:
            from factor_library.universe import get_universe_stats
            u = get_universe_stats()
            lines.append(f"\nUniverse: {u.get('active', 0)} 只活跃股票")
        except Exception:
            pass

        try:
            from data.trade_store import TradeStore
            store = TradeStore()
            open_opts = store.query_option_trades(status="open")
            lines.append(f"\n期权持仓: {len(open_opts)} 个 open")
            store.close()
        except Exception:
            pass

        return "\n".join(lines)
