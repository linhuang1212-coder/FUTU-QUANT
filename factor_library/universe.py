"""
全美股 Ticker Universe 管理

数据源优先级:
  1. GitHub rreichel3/US-Stock-Symbols (每日更新, JSON)
  2. NASDAQ screener CSV (备用)
  3. 本地缓存 (离线)

过滤规则:
  - 只保留普通股 (Common Stock), 排除 ETF/优先股/权证/ADR
  - 需要有最近交易数据 (非停牌/退市)
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from utils.logger import setup_logger

logger = setup_logger("factor_library.universe")

_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _ROOT / "data_store" / "universe" / "universe.db"
CACHE_JSON = _ROOT / "data_store" / "universe" / "tickers_raw.json"

GITHUB_URLS = {
    "nyse": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_full_tickers.json",
    "nasdaq": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json",
    "amex": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_full_tickers.json",
}


def _init_db(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS universe (
            symbol       TEXT PRIMARY KEY,
            name         TEXT,
            exchange     TEXT,
            sector       TEXT DEFAULT '',
            industry     TEXT DEFAULT '',
            market_cap   REAL DEFAULT 0,
            avg_volume   REAL DEFAULT 0,
            is_active    INTEGER DEFAULT 1,
            last_updated TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_universe_exchange ON universe(exchange);
        CREATE INDEX IF NOT EXISTS idx_universe_sector ON universe(sector);
        CREATE INDEX IF NOT EXISTS idx_universe_active ON universe(is_active);

        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol         TEXT,
            report_date    TEXT,
            pe             REAL,
            pb             REAL,
            roe            REAL,
            revenue_growth REAL,
            gross_margin   REAL,
            debt_equity    REAL,
            fcf_yield      REAL,
            dividend_yield REAL,
            piotroski_f    INTEGER,
            PRIMARY KEY (symbol, report_date)
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()


def fetch_tickers_github() -> list[dict]:
    """从 GitHub 下载全美股 ticker 列表."""
    all_tickers = []
    for exchange, url in GITHUB_URLS.items():
        logger.info(f"[Universe] 下载 {exchange.upper()} tickers ...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                item["exchange"] = exchange.upper()
            all_tickers.extend(data)
            logger.info(f"  {exchange.upper()}: {len(data)} tickers")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  {exchange.upper()} 下载失败: {e}")

    if all_tickers:
        CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
        CACHE_JSON.write_text(json.dumps(all_tickers, indent=2), encoding="utf-8")
        logger.info(f"[Universe] 缓存到 {CACHE_JSON} ({len(all_tickers)} total)")

    return all_tickers


def load_cached_tickers() -> list[dict]:
    """从本地缓存加载 ticker 列表."""
    if not CACHE_JSON.exists():
        return []
    return json.loads(CACHE_JSON.read_text(encoding="utf-8"))


def filter_common_stocks(tickers: list[dict]) -> list[dict]:
    """过滤只保留普通股, 排除 ETF/优先股/权证等."""
    filtered = []
    exclude_suffixes = ("-W", "-U", "-R", ".W", ".U", ".R")
    exclude_keywords = ("warrant", "preferred", "right", "unit", "debenture")

    for t in tickers:
        symbol = t.get("symbol", "")
        name = (t.get("name", "") or "").lower()

        if not symbol or len(symbol) > 5:
            continue
        if any(symbol.endswith(s) for s in exclude_suffixes):
            continue
        if any(kw in name for kw in exclude_keywords):
            continue
        if not symbol.isalpha():
            continue

        filtered.append(t)

    logger.info(f"[Universe] 过滤: {len(tickers)} -> {len(filtered)} 只普通股")
    return filtered


def save_to_db(tickers: list[dict]) -> int:
    """保存 ticker 列表到 SQLite."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)

    now = datetime.now().isoformat()
    count = 0
    for t in tickers:
        symbol = t.get("symbol", "")
        if not symbol:
            continue
        conn.execute("""
            INSERT OR REPLACE INTO universe
            (symbol, name, exchange, last_updated)
            VALUES (?, ?, ?, ?)
        """, (symbol, t.get("name", ""), t.get("exchange", ""), now))
        count += 1

    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 ("last_universe_update", now))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 ("universe_count", str(count)))
    conn.commit()
    conn.close()
    logger.info(f"[Universe] 写入 {count} 只到 SQLite")
    return count


def get_active_symbols(min_volume: float = 0) -> list[str]:
    """从 SQLite 获取活跃 ticker 列表."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)

    if min_volume > 0:
        cur = conn.execute(
            "SELECT symbol FROM universe WHERE is_active = 1 AND avg_volume >= ? ORDER BY symbol",
            (min_volume,))
    else:
        cur = conn.execute(
            "SELECT symbol FROM universe WHERE is_active = 1 ORDER BY symbol")

    symbols = [row[0] for row in cur.fetchall()]
    conn.close()
    return symbols


def get_universe_stats() -> dict:
    """获取 universe 统计信息."""
    if not DB_PATH.exists():
        return {"total": 0, "active": 0}
    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)

    total = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM universe WHERE is_active = 1").fetchone()[0]
    by_exchange = {}
    for row in conn.execute(
            "SELECT exchange, COUNT(*) FROM universe WHERE is_active = 1 GROUP BY exchange"):
        by_exchange[row[0]] = row[1]

    meta = {}
    for row in conn.execute("SELECT key, value FROM meta"):
        meta[row[0]] = row[1]

    conn.close()
    return {"total": total, "active": active, "by_exchange": by_exchange, "meta": meta}


def update_symbol_info(symbol: str, sector: str = "", industry: str = "",
                       market_cap: float = 0, avg_volume: float = 0):
    """更新单个 symbol 的元数据."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)
    conn.execute("""
        UPDATE universe SET sector = ?, industry = ?, market_cap = ?,
        avg_volume = ?, last_updated = ?
        WHERE symbol = ?
    """, (sector, industry, market_cap, avg_volume,
          datetime.now().isoformat(), symbol))
    conn.commit()
    conn.close()


def batch_update_info(info_list: list[dict]):
    """批量更新 symbol 元数据."""
    if not DB_PATH.exists() or not info_list:
        return
    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)
    now = datetime.now().isoformat()
    for info in info_list:
        conn.execute("""
            UPDATE universe SET sector = ?, industry = ?, market_cap = ?,
            avg_volume = ?, last_updated = ?
            WHERE symbol = ?
        """, (info.get("sector", ""), info.get("industry", ""),
              info.get("market_cap", 0), info.get("avg_volume", 0),
              now, info["symbol"]))
    conn.commit()
    conn.close()
    logger.info(f"[Universe] 批量更新 {len(info_list)} 只元数据")


def refresh_universe() -> dict:
    """完整刷新流程: 下载 -> 过滤 -> 入库."""
    logger.info("[Universe] 开始刷新全美股 ticker 列表...")
    raw = fetch_tickers_github()
    if not raw:
        logger.warning("[Universe] GitHub 下载失败, 尝试本地缓存")
        raw = load_cached_tickers()
    if not raw:
        logger.error("[Universe] 无法获取 ticker 列表")
        return {"total": 0, "filtered": 0, "saved": 0}

    filtered = filter_common_stocks(raw)
    saved = save_to_db(filtered)
    stats = get_universe_stats()

    logger.info(f"[Universe] 刷新完成: 原始={len(raw)}, 过滤后={len(filtered)}, 入库={saved}")
    return {"total": len(raw), "filtered": len(filtered), "saved": saved, "stats": stats}
