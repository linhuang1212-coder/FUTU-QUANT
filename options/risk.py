from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("options.risk")

# Correlated asset groups — same-group positions count as correlated exposure
CORRELATED_GROUPS: dict[str, set[str]] = {
    "US.SPY": {"US.QQQ", "US.IWM", "US.DIA", "US.TQQQ"},
    "US.QQQ": {"US.SPY", "US.IWM", "US.DIA", "US.TQQQ", "US.AAPL", "US.MSFT", "US.NVDA"},
    "US.IWM": {"US.SPY", "US.QQQ", "US.DIA"},
    "US.DIA": {"US.SPY", "US.QQQ", "US.IWM"},
    "US.TQQQ": {"US.SPY", "US.QQQ", "US.SOXL"},
    "US.SOXL": {"US.TQQQ", "US.QQQ", "US.AMD", "US.NVDA"},
    "US.META": {"US.GOOGL", "US.AMZN"},
    "US.GOOGL": {"US.META", "US.AMZN"},
    "US.AMZN": {"US.META", "US.GOOGL"},
    "US.AAPL": {"US.QQQ", "US.MSFT"},
    "US.MSFT": {"US.QQQ", "US.AAPL"},
    "US.NVDA": {"US.QQQ", "US.AMD", "US.SOXL"},
    "US.AMD": {"US.NVDA", "US.SOXL"},
    "US.TSLA": set(),
    "US.XLK": {"US.QQQ"},
    "US.XLF": set(),
    "US.XLE": set(),
    "US.XLV": set(),
    "US.GLD": set(),
    "US.TLT": set(),
}


class OptionsRiskManager:
    """Per-strategy + portfolio-level risk management with PDT guard."""

    # Strategies that are intraday (same-day open+close) consume PDT quota
    INTRADAY_STRATEGIES = {"orb", "orb_0dte"}

    def __init__(self, config: dict, pdt_guard=None):
        alloc = config.get("capital_allocation", {})
        self.capital = sum(alloc.values())
        self.allocations = alloc

        risk = config.get("risk", {})
        self.max_daily_loss = float(risk.get("max_daily_loss", self.capital * 0.10))
        self.max_monthly_dd = float(risk.get("max_monthly_drawdown", self.capital * 0.20))

        portfolio = risk.get("portfolio", {})
        self.max_portfolio_risk_pct = float(portfolio.get("max_total_risk_pct", 0.20))
        self.max_single_exposure_pct = float(portfolio.get("max_single_exposure_pct", 0.15))
        self.max_correlated_positions = int(portfolio.get("max_correlated_positions", 1))

        self._per_strategy: dict[str, dict] = {}
        for key, params in risk.get("per_strategy", {}).items():
            if isinstance(params, dict):
                self._per_strategy[key] = params

        self._daily_pnl: float = 0.0
        self._daily_date: str = ""
        self._monthly_pnl: float = 0.0
        self._monthly_key: str = ""
        self._open_capital: dict[str, float] = {}
        self._daily_trade_count: dict[str, int] = {}

        # Portfolio-level tracking
        self._open_positions: list[dict] = []  # {underlying, max_loss, strategy}

        # PDT guard — shared across stock + option systems
        self._pdt = pdt_guard

    def _reset_daily_if_needed(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self._daily_date != today:
            self._daily_date = today
            self._daily_pnl = 0.0
            self._daily_trade_count.clear()

    def _reset_monthly_if_needed(self):
        month_key = datetime.now().strftime("%Y-%m")
        if self._monthly_key != month_key:
            self._monthly_key = month_key
            self._monthly_pnl = 0.0

    def _strategy_key(self, strategy: str) -> str:
        mapping = {
            "orb_0dte": "orb",
            "earnings_spread": "earnings",
            "wheel_csp": "wheel",
            "wheel_cc": "wheel",
            "pmcc_leaps": "pmcc",
            "pmcc_short": "pmcc",
        }
        return mapping.get(strategy, strategy)

    def is_strategy_enabled(self, strategy: str) -> bool:
        key = self._strategy_key(strategy)
        params = self._per_strategy.get(key, {})
        return params.get("enabled", True)

    def can_open_trade(self, strategy: str, premium: float,
                       underlying: str = "", max_loss: float = 0) -> tuple[bool, str]:
        self._reset_daily_if_needed()
        self._reset_monthly_if_needed()

        key = self._strategy_key(strategy)
        if not self.is_strategy_enabled(strategy):
            return False, f"Strategy '{key}' is disabled"

        # ── PDT check for intraday strategies ──
        if key in self.INTRADAY_STRATEGIES and self._pdt is not None:
            if not self._pdt.can_day_trade():
                remaining = self._pdt.remaining_day_trades()
                return False, (f"[PDT] No day trades remaining "
                               f"({remaining}/3 in rolling 5-day window)")
            # Reserve 1 PDT slot as emergency buffer
            max_weekly = self._per_strategy.get(key, {}).get("max_trades_per_week")
            if max_weekly is not None:
                used = self._pdt._recent_count()
                if used >= int(max_weekly):
                    return False, (f"[PDT] Weekly limit reached for '{key}' "
                                   f"({used}/{max_weekly} this week)")
            if self._pdt.should_warn():
                logger.warning(f"[PDT] Last day trade available — use with caution")

        # ── Per-strategy checks ──
        params = self._per_strategy.get(key, {})
        max_per_trade = float(params.get("max_loss_per_trade", self.capital * 0.05))
        if premium > max_per_trade:
            return False, (f"Premium ${premium:.0f} exceeds {key} "
                           f"per-trade limit ${max_per_trade:.0f}")

        max_daily_trades = params.get("max_trades_per_day")
        if max_daily_trades is not None:
            count = self._daily_trade_count.get(key, 0)
            if count >= int(max_daily_trades):
                return False, (f"Strategy '{key}' daily trade limit "
                               f"({count}/{max_daily_trades})")

        if self._daily_pnl <= -self.max_daily_loss:
            return False, (f"Daily loss limit reached "
                           f"(${self._daily_pnl:.0f}/${-self.max_daily_loss:.0f})")

        if self._monthly_pnl <= -self.max_monthly_dd:
            return False, f"Monthly drawdown limit reached (${self._monthly_pnl:.0f})"

        strategy_alloc = self.allocations.get(strategy, 0)
        committed = self._open_capital.get(strategy, 0)
        if strategy_alloc > 0 and committed + premium > strategy_alloc:
            return False, (f"Strategy '{strategy}' allocation exhausted: "
                           f"${committed:.0f} + ${premium:.0f} > ${strategy_alloc:.0f}")

        # ── Portfolio-level checks ──
        trade_risk = max_loss if max_loss > 0 else premium
        ok, reason = self._check_portfolio_risk(underlying, trade_risk)
        if not ok:
            return False, reason

        return True, "OK"

    def _check_portfolio_risk(self, underlying: str, new_max_loss: float) -> tuple[bool, str]:
        """Portfolio-level risk checks before opening a new position."""
        if not underlying:
            return True, "OK"

        # 1. Total portfolio max loss must not exceed monthly drawdown limit
        total_risk = sum(p["max_loss"] for p in self._open_positions) + new_max_loss
        limit = self.capital * self.max_portfolio_risk_pct
        if total_risk > limit:
            return False, (f"[PORTFOLIO] Total risk ${total_risk:.0f} "
                           f"exceeds {self.max_portfolio_risk_pct:.0%} limit "
                           f"${limit:.0f} (monthly DD=${self.max_monthly_dd:.0f})")

        # 2. Single underlying exposure cap
        exposure_pct = new_max_loss / self.capital
        if exposure_pct > self.max_single_exposure_pct:
            return False, (f"[PORTFOLIO] {underlying} exposure "
                           f"{exposure_pct:.1%} exceeds "
                           f"{self.max_single_exposure_pct:.0%} cap")

        # 3. Correlated position limit
        correlated = CORRELATED_GROUPS.get(underlying, set())
        sym_clean = underlying.replace("US.", "")
        correlated_count = 0
        correlated_names = []
        for p in self._open_positions:
            p_sym = p["underlying"]
            if p_sym == underlying:
                correlated_count += 1
                correlated_names.append(p_sym)
            elif p_sym in correlated:
                correlated_count += 1
                correlated_names.append(p_sym)

        if correlated_count >= self.max_correlated_positions:
            return False, (f"[PORTFOLIO] {underlying} has "
                           f"{correlated_count} correlated position(s) "
                           f"already open: {correlated_names}")

        return True, "OK"

    def on_trade_open(self, strategy: str, premium: float,
                      underlying: str = "", max_loss: float = 0):
        self._reset_daily_if_needed()
        key = self._strategy_key(strategy)
        self._open_capital[strategy] = self._open_capital.get(strategy, 0) + premium
        self._daily_trade_count[key] = self._daily_trade_count.get(key, 0) + 1
        if underlying:
            self._open_positions.append({
                "underlying": underlying,
                "max_loss": max_loss if max_loss > 0 else premium,
                "strategy": strategy,
            })
        # Record PDT usage for intraday strategies
        if key in self.INTRADAY_STRATEGIES and self._pdt is not None:
            self._pdt.record_day_trade(underlying or strategy)
            remaining = self._pdt.remaining_day_trades()
            logger.info(f"[PDT] Day trade recorded for {underlying}. "
                        f"Remaining: {remaining}/3")

    def on_trade_close(self, strategy: str, premium: float, pnl: float,
                       underlying: str = ""):
        self._reset_daily_if_needed()
        self._reset_monthly_if_needed()
        self._open_capital[strategy] = max(0, self._open_capital.get(strategy, 0) - premium)
        self._daily_pnl += pnl
        self._monthly_pnl += pnl
        if underlying:
            self._open_positions = [
                p for p in self._open_positions
                if not (p["underlying"] == underlying and p["strategy"] == strategy)
            ]
        if pnl < 0:
            logger.warning(f"[RISK] Loss ${pnl:.2f} on {strategy}. "
                           f"Daily: ${self._daily_pnl:.2f}, "
                           f"Monthly: ${self._monthly_pnl:.2f}")

    def get_status(self) -> dict:
        self._reset_daily_if_needed()
        self._reset_monthly_if_needed()
        total_risk = sum(p["max_loss"] for p in self._open_positions)
        status = {
            "capital": self.capital,
            "daily_pnl": self._daily_pnl,
            "monthly_pnl": self._monthly_pnl,
            "open_positions": dict(self._open_capital),
            "daily_limit": self.max_daily_loss,
            "monthly_limit": self.max_monthly_dd,
            "daily_trade_counts": dict(self._daily_trade_count),
            "portfolio_total_risk": total_risk,
            "portfolio_risk_limit": self.capital * self.max_portfolio_risk_pct,
            "positions": list(self._open_positions),
        }
        if self._pdt is not None:
            status["pdt_remaining"] = self._pdt.remaining_day_trades()
            status["pdt_max"] = self._pdt.max_day_trades
        return status


ETF_SYMBOLS = {
    "US.SPY", "US.QQQ", "US.IWM", "US.DIA",
    "US.XLK", "US.XLF", "US.XLE", "US.XLV",
    "US.TQQQ", "US.SOXL", "US.GLD", "US.TLT",
}


class EarningsGuard:
    """Block option trades when earnings fall within the option's expiry.

    Data sources tried in order:
    1. Financial Modeling Prep (FMP) free API
    2. Futu snapshot fields (unreliable for earnings)
    3. Conservative fallback: individual stocks with unknown date → BLOCK
    """

    FMP_URL = "https://financialmodelingprep.com/stable/earnings-calendar"

    def __init__(self, quote_ctx=None, fmp_api_key: str = ""):
        self._ctx = quote_ctx
        self._fmp_key = fmp_api_key
        self._cache: dict[str, Optional[str]] = {}

    def get_next_earnings(self, symbol: str) -> Optional[str]:
        """Get next earnings date for symbol. Returns YYYY-MM-DD or None."""
        if symbol in self._cache:
            return self._cache[symbol]

        if symbol in ETF_SYMBOLS:
            return None  # ETFs don't report earnings

        ticker = symbol.replace("US.", "")

        # Source 1: FMP API
        date = self._fetch_fmp(ticker)
        if date:
            self._cache[symbol] = date
            return date

        # Source 2: Futu snapshot (best-effort)
        date = self._fetch_futu(symbol)
        if date:
            self._cache[symbol] = date
            return date

        # Not found
        self._cache[symbol] = None
        return None

    def _fetch_fmp(self, ticker: str) -> Optional[str]:
        if not self._fmp_key:
            return None
        try:
            import requests
            from datetime import date, timedelta
            today = date.today()
            params = {
                "symbol": ticker,
                "from": today.isoformat(),
                "to": (today + timedelta(days=90)).isoformat(),
                "apikey": self._fmp_key,
            }
            resp = requests.get(self.FMP_URL, params=params, timeout=10)
            if resp.status_code != 200:
                logger.debug(f"[EARNINGS] FMP returned {resp.status_code} for {ticker}")
                return None
            data = resp.json()
            if isinstance(data, list) and data:
                d = data[0].get("date", "")[:10]
                logger.info(f"[EARNINGS] {ticker} next earnings: {d} (FMP)")
                return d
        except Exception as e:
            logger.debug(f"[EARNINGS] FMP lookup failed for {ticker}: {e}")
        return None

    def _fetch_futu(self, symbol: str) -> Optional[str]:
        if self._ctx is None:
            return None
        try:
            from futu import RET_OK
            import time
            time.sleep(0.3)
            ret, data = self._ctx.get_market_snapshot([symbol])
            if ret == RET_OK and data is not None and len(data) > 0:
                row = data.iloc[0]
                for col in ("earnings_time", "next_earnings", "earnings_date"):
                    if col in row and row[col]:
                        d = str(row[col])[:10]
                        logger.info(f"[EARNINGS] {symbol} next earnings: {d} (Futu)")
                        return d
        except Exception:
            pass
        return None

    def check(self, symbol: str, expiry: str) -> tuple[bool, str]:
        """Check if it's safe to open an option position.

        Returns (safe, reason). For individual stocks with unknown earnings
        dates, the conservative default is to BLOCK the trade.
        """
        if symbol in ETF_SYMBOLS:
            return True, "ETF — no earnings risk"

        earnings = self.get_next_earnings(symbol)

        if earnings is None:
            # Unknown earnings = block individual stocks
            logger.warning(f"[EARNINGS BLOCK] {symbol} earnings date unknown "
                           f"— blocking individual stock")
            return False, (f"{symbol} earnings date unknown — "
                           f"individual stocks blocked without calendar data")

        try:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            earn_dt = datetime.strptime(earnings, "%Y-%m-%d")
            if earn_dt <= exp_dt:
                logger.warning(f"[EARNINGS BLOCK] {symbol} earnings {earnings} "
                               f"before expiry {expiry}")
                return False, (f"{symbol} earnings {earnings} is before "
                               f"expiry {expiry} — blocked")
            days_margin = (earn_dt - exp_dt).days
            logger.info(f"[EARNINGS OK] {symbol} earnings {earnings} is "
                        f"{days_margin}d after expiry {expiry}")
        except ValueError:
            return False, f"{symbol} date parse error — blocked"

        return True, "OK"
