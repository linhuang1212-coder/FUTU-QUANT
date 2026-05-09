"""
Paper Trading Engine — 模拟盘验证系统

对 value / low_risk / momentum 三个因子模型进行虚拟交易，
每月调仓一次，记录净值曲线，与回测结果对比偏差。

数据存储在 data_store/paper_trading.db (SQLite)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("paper_trading")

DB_PATH = Path(__file__).resolve().parent.parent / "data_store" / "paper_trading.db"

MODELS = ["value", "low_risk", "momentum"]
DEFAULT_CAPITAL = 10_000.0
TOP_N = 20


class PaperTrader:
    """Virtual multi-model portfolio tracker backed by SQLite."""

    def __init__(self, db_path: Path = DB_PATH, capital: float = DEFAULT_CAPITAL):
        self.db_path = db_path
        self.initial_capital = capital
        self._conn: Optional[sqlite3.Connection] = None

    # ── DB lifecycle ──

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._init_tables()
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_tables(self):
        c = self._connect()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS portfolios (
                model       TEXT NOT NULL,
                capital     REAL NOT NULL,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (model)
            );
            CREATE TABLE IF NOT EXISTS holdings (
                model       TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                shares      REAL NOT NULL,
                cost_price  REAL NOT NULL,
                entry_date  TEXT NOT NULL,
                PRIMARY KEY (model, symbol)
            );
            CREATE TABLE IF NOT EXISTS nav_history (
                model       TEXT NOT NULL,
                date        TEXT NOT NULL,
                nav         REAL NOT NULL,
                cash        REAL NOT NULL,
                n_holdings  INTEGER NOT NULL,
                PRIMARY KEY (model, date)
            );
            CREATE TABLE IF NOT EXISTS rebalance_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model       TEXT NOT NULL,
                date        TEXT NOT NULL,
                buys        TEXT,
                sells       TEXT,
                turnover    REAL,
                nav_before  REAL,
                nav_after   REAL
            );
        """)
        c.commit()

    # ── Init ──

    def init_portfolios(self):
        """Create portfolio records for each model if they don't exist."""
        c = self._connect()
        now = datetime.now().isoformat()
        for model in MODELS:
            existing = c.execute(
                "SELECT model FROM portfolios WHERE model=?", (model,)).fetchone()
            if not existing:
                c.execute("INSERT INTO portfolios VALUES (?,?,?)",
                          (model, self.initial_capital, now))
                logger.info(f"[INIT] {model} portfolio created, "
                            f"capital=${self.initial_capital:,.0f}")
        c.commit()

    # ── Factor library integration ──

    def _load_factor_matrix(self) -> pd.DataFrame:
        from factor_library.storage import load_factors
        from factor_library.search import build_factor_matrix

        categories = ["technical", "risk", "volatility", "liquidity", "fundamental"]
        factor_dfs = {}
        for cat in categories:
            df = load_factors(cat)
            if not df.empty:
                factor_dfs[cat] = df
        if not factor_dfs:
            return pd.DataFrame()
        return build_factor_matrix(factor_dfs)

    def _get_target_holdings(self, model: str,
                             matrix: pd.DataFrame) -> list[str]:
        """Score stocks and return top-N symbols for given model."""
        from factor_library.screener import score_stocks, risk_filter

        filtered = risk_filter(matrix, top_pct=0.8)
        if filtered.empty:
            filtered = matrix

        results = score_stocks(filtered, model=model, top_n=TOP_N)
        if results.empty:
            return []
        return results["symbol"].tolist()

    # ── Weight optimization ──

    @staticmethod
    def _compute_weights(symbols: list[str]) -> dict[str, float]:
        """Compute HRP weights, fallback to equal weight."""
        if len(symbols) < 2:
            return {s: 1.0 for s in symbols}
        try:
            from factor_library.optimizer import compute_weights
            weights = compute_weights(symbols, method="hrp", period="1y")
            logger.info(f"  [HRP] Computed weights for {len(symbols)} stocks")
            return weights
        except Exception as e:
            logger.warning(f"  [HRP] Failed ({e}), using equal weight")
            w = 1.0 / len(symbols)
            return {s: w for s in symbols}

    # ── Price fetching ──

    @staticmethod
    def _fetch_prices(symbols: list[str]) -> dict[str, float]:
        """Fetch latest prices via yfinance."""
        if not symbols:
            return {}
        try:
            import yfinance as yf
            tickers = " ".join(symbols)
            data = yf.download(tickers, period="2d", progress=False)
            if data.empty:
                return {}
            prices = {}
            if len(symbols) == 1:
                close = data["Close"]
                if not close.empty:
                    prices[symbols[0]] = float(close.iloc[-1])
            else:
                close = data["Close"]
                for sym in symbols:
                    if sym in close.columns and not pd.isna(close[sym].iloc[-1]):
                        prices[sym] = float(close[sym].iloc[-1])
            return prices
        except Exception as e:
            logger.error(f"Price fetch failed: {e}")
            return {}

    # ── Portfolio operations ──

    def get_cash(self, model: str) -> float:
        c = self._connect()
        row = c.execute(
            "SELECT capital FROM portfolios WHERE model=?", (model,)).fetchone()
        return float(row["capital"]) if row else 0.0

    def _set_cash(self, model: str, cash: float):
        c = self._connect()
        c.execute("UPDATE portfolios SET capital=? WHERE model=?", (cash, model))

    def get_holdings(self, model: str) -> list[dict]:
        c = self._connect()
        rows = c.execute(
            "SELECT * FROM holdings WHERE model=?", (model,)).fetchall()
        return [dict(r) for r in rows]

    def _compute_nav(self, model: str, prices: dict[str, float]) -> float:
        cash = self.get_cash(model)
        holdings = self.get_holdings(model)
        mkt_val = sum(
            h["shares"] * prices.get(h["symbol"], h["cost_price"])
            for h in holdings
        )
        return cash + mkt_val

    # ── Rebalancing ──

    def rebalance(self, model: str):
        """Run a full rebalance for one model."""
        matrix = self._load_factor_matrix()
        if matrix.empty:
            logger.warning(f"[{model}] Factor matrix empty, skip rebalance")
            return

        target_symbols = self._get_target_holdings(model, matrix)
        if not target_symbols:
            logger.warning(f"[{model}] No target holdings, skip rebalance")
            return

        current = self.get_holdings(model)
        current_syms = {h["symbol"] for h in current}

        all_syms = list(current_syms | set(target_symbols))
        prices = self._fetch_prices(all_syms)
        if not prices:
            logger.error(f"[{model}] Cannot fetch prices, abort rebalance")
            return

        nav_before = self._compute_nav(model, prices)

        # Sell everything not in target
        sells = []
        c = self._connect()
        cash = self.get_cash(model)
        for h in current:
            sym = h["symbol"]
            if sym not in target_symbols:
                price = prices.get(sym, h["cost_price"])
                proceeds = h["shares"] * price
                cash += proceeds
                sells.append({"symbol": sym, "shares": h["shares"],
                              "price": price})
                c.execute("DELETE FROM holdings WHERE model=? AND symbol=?",
                          (model, sym))
                logger.info(f"  [{model}] SELL {sym} {h['shares']:.2f} sh "
                            f"@ ${price:.2f}")

        # HRP-weighted buy into target
        valid_targets = [s for s in target_symbols if s in prices]
        if not valid_targets:
            self._set_cash(model, cash)
            c.commit()
            return

        weights = self._compute_weights(valid_targets)

        buys = []
        remaining_cash = cash

        for sym in valid_targets:
            price = prices[sym]
            if price <= 0:
                continue
            w = weights.get(sym, 1.0 / len(valid_targets))
            allocation = cash * w
            shares = allocation / price
            cost = shares * price
            remaining_cash -= cost
            buys.append({"symbol": sym, "shares": shares, "price": price})

            existing = c.execute(
                "SELECT shares, cost_price FROM holdings WHERE model=? AND symbol=?",
                (model, sym)).fetchone()
            if existing:
                total_shares = existing["shares"] + shares
                avg_cost = ((existing["shares"] * existing["cost_price"])
                            + cost) / total_shares
                c.execute(
                    "UPDATE holdings SET shares=?, cost_price=? "
                    "WHERE model=? AND symbol=?",
                    (total_shares, avg_cost, model, sym))
            else:
                c.execute(
                    "INSERT INTO holdings VALUES (?,?,?,?,?)",
                    (model, sym, shares, price, date.today().isoformat()))

            logger.info(f"  [{model}] BUY  {sym} {shares:.2f} sh @ ${price:.2f}")

        self._set_cash(model, max(remaining_cash, 0))
        c.commit()

        nav_after = self._compute_nav(model, prices)
        turnover = (sum(b["shares"] * b["price"] for b in buys)
                    + sum(s["shares"] * s["price"] for s in sells))

        c.execute(
            "INSERT INTO rebalance_log (model,date,buys,sells,turnover,"
            "nav_before,nav_after) VALUES (?,?,?,?,?,?,?)",
            (model, date.today().isoformat(),
             json.dumps([b["symbol"] for b in buys]),
             json.dumps([s["symbol"] for s in sells]),
             turnover, nav_before, nav_after))
        c.commit()

        logger.info(f"[{model}] Rebalance done: "
                    f"{len(sells)} sells, {len(buys)} buys, "
                    f"NAV ${nav_before:.0f} -> ${nav_after:.0f}")

    def rebalance_all(self):
        for model in MODELS:
            logger.info(f"{'='*40}")
            logger.info(f"Rebalancing {model}...")
            self.rebalance(model)

    # ── Daily NAV snapshot ──

    def record_daily_nav(self):
        """Fetch prices and record NAV for all models."""
        all_symbols = set()
        for model in MODELS:
            for h in self.get_holdings(model):
                all_symbols.add(h["symbol"])

        prices = self._fetch_prices(list(all_symbols))
        if not prices and all_symbols:
            logger.warning("Cannot fetch prices for NAV snapshot")
            return

        c = self._connect()
        today = date.today().isoformat()
        for model in MODELS:
            nav = self._compute_nav(model, prices)
            holdings = self.get_holdings(model)
            c.execute(
                "INSERT OR REPLACE INTO nav_history VALUES (?,?,?,?,?)",
                (model, today, nav, self.get_cash(model), len(holdings)))
            logger.info(f"[{model}] NAV=${nav:,.2f} "
                        f"({len(holdings)} holdings, "
                        f"cash=${self.get_cash(model):,.2f})")
        c.commit()

    # ── Reporting ──

    def get_nav_series(self, model: str) -> pd.DataFrame:
        c = self._connect()
        rows = c.execute(
            "SELECT date, nav FROM nav_history WHERE model=? ORDER BY date",
            (model,)).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def generate_report(self) -> str:
        """Generate a comparison report across all models."""
        lines = ["Paper Trading Report", "=" * 40, ""]

        for model in MODELS:
            nav_df = self.get_nav_series(model)
            holdings = self.get_holdings(model)
            cash = self.get_cash(model)

            lines.append(f"Model: {model.upper()}")
            lines.append(f"  Holdings: {len(holdings)}")
            lines.append(f"  Cash: ${cash:,.2f}")

            if not nav_df.empty:
                latest_nav = nav_df["nav"].iloc[-1]
                returns = (latest_nav / self.initial_capital - 1) * 100
                lines.append(f"  NAV: ${latest_nav:,.2f} ({returns:+.2f}%)")

                if len(nav_df) > 1:
                    daily_ret = nav_df["nav"].pct_change().dropna()
                    sharpe = (daily_ret.mean() / daily_ret.std()
                              * np.sqrt(252)) if daily_ret.std() > 0 else 0
                    max_dd = ((nav_df["nav"] / nav_df["nav"].cummax()) - 1).min()
                    lines.append(f"  Sharpe: {sharpe:.2f}")
                    lines.append(f"  Max DD: {max_dd:.2%}")
            else:
                lines.append(f"  NAV: ${self.initial_capital:,.2f} (no history)")
            lines.append("")

        # Rebalance history
        c = self._connect()
        recent = c.execute(
            "SELECT * FROM rebalance_log ORDER BY date DESC LIMIT 5").fetchall()
        if recent:
            lines.append("Recent Rebalances:")
            for r in recent:
                lines.append(f"  {r['date']} {r['model']}: "
                             f"NAV ${r['nav_before']:.0f} -> ${r['nav_after']:.0f}")

        return "\n".join(lines)

    def generate_telegram_report(self) -> str:
        """Generate HTML-formatted report for Telegram."""
        lines = ["<b>Paper Trading 周报</b>\n"]

        for model in MODELS:
            nav_df = self.get_nav_series(model)
            holdings = self.get_holdings(model)
            cash = self.get_cash(model)

            if not nav_df.empty:
                latest_nav = nav_df["nav"].iloc[-1]
                returns = (latest_nav / self.initial_capital - 1) * 100
                emoji = "+" if returns >= 0 else ""

                lines.append(f"<b>{model.upper()}</b>")
                lines.append(f"  NAV: ${latest_nav:,.2f} ({emoji}{returns:.2f}%)")
                lines.append(f"  Holdings: {len(holdings)} | Cash: ${cash:,.2f}")

                if len(nav_df) > 1:
                    daily_ret = nav_df["nav"].pct_change().dropna()
                    sharpe = (daily_ret.mean() / daily_ret.std()
                              * np.sqrt(252)) if daily_ret.std() > 0 else 0
                    max_dd = ((nav_df["nav"] / nav_df["nav"].cummax()) - 1).min()
                    lines.append(f"  Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}")
            else:
                lines.append(f"<b>{model.upper()}</b>")
                lines.append(f"  NAV: ${self.initial_capital:,.2f} (no trades)")
            lines.append("")

        return "\n".join(lines)
