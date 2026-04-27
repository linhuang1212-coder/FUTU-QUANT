import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class TradeStore:
    """SQLite-backed persistent storage for trades and PDT day-trade records."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            root = Path(__file__).resolve().parent.parent
            db_dir = root / "data_store"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "trades.db")

        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                action    TEXT    NOT NULL,
                symbol    TEXT    NOT NULL,
                strategy  TEXT    DEFAULT '',
                qty       INTEGER NOT NULL,
                price     REAL    NOT NULL,
                pnl       REAL    DEFAULT 0,
                pdt_remaining INTEGER DEFAULT 3,
                dry_run   INTEGER DEFAULT 0,
                error     TEXT    DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp);

            CREATE TABLE IF NOT EXISTS pdt_day_trades (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT    NOT NULL,
                timestamp TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pdt_timestamp
                ON pdt_day_trades(timestamp);

            CREATE TABLE IF NOT EXISTS option_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                strategy        TEXT    NOT NULL,
                underlying      TEXT    NOT NULL,
                legs            TEXT    NOT NULL,
                max_loss        REAL,
                target_pnl      REAL,
                status          TEXT    DEFAULT 'open',
                close_timestamp TEXT,
                realized_pnl    REAL    DEFAULT 0,
                close_reason    TEXT    DEFAULT '',
                dry_run         INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_opt_strategy
                ON option_trades(strategy);
            CREATE INDEX IF NOT EXISTS idx_opt_underlying
                ON option_trades(underlying);
            CREATE INDEX IF NOT EXISTS idx_opt_status
                ON option_trades(status);
        """)
        self._conn.commit()

    # ── Trade logging ────────────────────────────────────────────

    def log_trade(
        self,
        action: str,
        symbol: str,
        qty: int,
        price: float,
        pnl: float = 0.0,
        strategy: str = "",
        pdt_remaining: int = 3,
        dry_run: bool = False,
        error: str = "",
    ) -> int:
        """Insert a trade record. Returns the row id."""
        ts = datetime.now().isoformat()
        cur = self._conn.execute(
            """INSERT INTO trades
               (timestamp, action, symbol, strategy, qty, price,
                pnl, pdt_remaining, dry_run, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, action, symbol, strategy, qty, price,
             round(pnl, 2), pdt_remaining, int(dry_run), error),
        )
        self._conn.commit()
        return cur.lastrowid

    def query_trades(
        self,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 200,
    ) -> list[dict]:
        """Query trades with optional filters."""
        clauses: list[str] = []
        params: list = []

        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            clauses.append("timestamp >= ?")
            params.append(cutoff)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        sql = f"SELECT * FROM trades {where} ORDER BY id DESC LIMIT ?"

        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── PDT day-trade records ────────────────────────────────────

    def record_day_trade(self, symbol: str) -> None:
        ts = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO pdt_day_trades (symbol, timestamp) VALUES (?, ?)",
            (symbol, ts),
        )
        self._conn.commit()

    def get_recent_day_trades(self, days: int = 5) -> list[dict]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cur = self._conn.execute(
            "SELECT symbol, timestamp FROM pdt_day_trades WHERE timestamp >= ? ORDER BY id",
            (cutoff,),
        )
        return [{"symbol": row[0], "timestamp": row[1]} for row in cur.fetchall()]

    def count_recent_day_trades(self, days: int = 5) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM pdt_day_trades WHERE timestamp >= ?",
            (cutoff,),
        )
        return cur.fetchone()[0]

    def cleanup_old_day_trades(self, days: int = 5) -> int:
        """Delete day-trade records older than `days`. Returns rows deleted."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cur = self._conn.execute(
            "DELETE FROM pdt_day_trades WHERE timestamp < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Option trade records ─────────────────────────────────────

    def log_option_trade(
        self,
        strategy: str,
        underlying: str,
        legs_json: str,
        max_loss: float = 0.0,
        target_pnl: float = 0.0,
        status: str = "open",
        dry_run: bool = False,
    ) -> int:
        ts = datetime.now().isoformat()
        cur = self._conn.execute(
            """INSERT INTO option_trades
               (timestamp, strategy, underlying, legs, max_loss,
                target_pnl, status, dry_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, strategy, underlying, legs_json,
             round(max_loss, 2), round(target_pnl, 2), status, int(dry_run)),
        )
        self._conn.commit()
        return cur.lastrowid

    def close_option_trade(
        self,
        trade_id: int,
        realized_pnl: float,
        close_reason: str = "",
    ) -> None:
        ts = datetime.now().isoformat()
        self._conn.execute(
            """UPDATE option_trades
               SET status = 'closed', close_timestamp = ?,
                   realized_pnl = ?, close_reason = ?
               WHERE id = ?""",
            (ts, round(realized_pnl, 2), close_reason, trade_id),
        )
        self._conn.commit()

    def query_option_trades(
        self,
        strategy: Optional[str] = None,
        underlying: Optional[str] = None,
        status: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if underlying:
            clauses.append("underlying = ?")
            params.append(underlying)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            clauses.append("timestamp >= ?")
            params.append(cutoff)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        sql = f"SELECT * FROM option_trades {where} ORDER BY id DESC LIMIT ?"
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Lifecycle ────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
