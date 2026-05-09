"""
Earnings Calendar — free data sources for upcoming earnings dates.

Sources (tried in order):
  1. SEC EDGAR XBRL filings (CIK-based, free, authoritative)
  2. Yahoo Finance earnings calendar (free, web scraping)
  3. Financial Modeling Prep (free tier: 250 req/day)

Caches results to SQLite to minimize API calls.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import requests

from utils.logger import setup_logger

logger = setup_logger("data.earnings_calendar")

_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _ROOT / "data_store" / "earnings.db"
_CACHE_DAYS = 1

SEC_HEADERS = {
    "User-Agent": "FUTU-QUANT Research bot (personal use)",
    "Accept": "application/json",
}


def _init_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings (
            symbol TEXT PRIMARY KEY,
            earnings_date TEXT,
            source TEXT,
            fetched_at TEXT
        )
    """)
    conn.commit()
    return conn


def _get_cached(symbol: str) -> Optional[str]:
    """Return cached earnings date if fresh enough."""
    try:
        conn = _init_db()
        row = conn.execute(
            "SELECT earnings_date, fetched_at FROM earnings WHERE symbol=?",
            (symbol,)).fetchone()
        conn.close()
        if row:
            fetched = datetime.fromisoformat(row[1])
            if (datetime.now() - fetched).days < _CACHE_DAYS:
                return row[0]
    except Exception:
        pass
    return None


def _save_cached(symbol: str, earnings_date: str, source: str):
    try:
        conn = _init_db()
        conn.execute("""
            INSERT OR REPLACE INTO earnings (symbol, earnings_date, source, fetched_at)
            VALUES (?, ?, ?, ?)
        """, (symbol, earnings_date, source, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_upcoming_earnings_yahoo(symbols: list[str],
                                days_ahead: int = 14) -> list[dict]:
    """Fetch upcoming earnings dates from Yahoo Finance.

    Uses the earnings calendar endpoint which is free and doesn't need auth.
    """
    results = []
    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    for batch_start in range(0, (end_date - today).days + 1, 7):
        d_from = today + timedelta(days=batch_start)
        d_to = min(d_from + timedelta(days=6), end_date)

        url = (f"https://finance.yahoo.com/calendar/earnings"
               f"?from={d_from.isoformat()}&to={d_to.isoformat()}")

        try:
            import re
            resp = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }, timeout=15)
            if resp.status_code != 200:
                continue

            # Parse ticker and date from HTML table
            text = resp.text
            symbol_set = set(s.replace("US.", "").upper() for s in symbols)

            rows = re.findall(
                r'data-symbol="([A-Z]+)".*?'
                r'aria-label="Earnings Date"[^>]*>([^<]+)<',
                text, re.DOTALL)

            for ticker, date_str in rows:
                if ticker.upper() in symbol_set:
                    try:
                        parsed = datetime.strptime(date_str.strip()[:12],
                                                    "%b %d, %Y")
                        earn_date = parsed.strftime("%Y-%m-%d")
                        results.append({
                            "symbol": f"US.{ticker.upper()}",
                            "earnings_date": earn_date,
                            "source": "yahoo",
                        })
                        _save_cached(f"US.{ticker.upper()}", earn_date, "yahoo")
                    except ValueError:
                        pass

            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"[Earnings] Yahoo calendar error: {e}")

    return results


def get_upcoming_earnings_fmp(symbols: list[str], api_key: str,
                               days_ahead: int = 14) -> list[dict]:
    """Fetch upcoming earnings from Financial Modeling Prep API."""
    if not api_key:
        return []

    results = []
    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    url = "https://financialmodelingprep.com/stable/earnings-calendar"
    try:
        resp = requests.get(url, params={
            "from": today.isoformat(),
            "to": end_date.isoformat(),
            "apikey": api_key,
        }, timeout=15)

        if resp.status_code != 200:
            return []

        data = resp.json()
        symbol_set = set(s.replace("US.", "").upper() for s in symbols)

        for entry in data:
            ticker = entry.get("symbol", "").upper()
            earn_date = entry.get("date", "")[:10]
            if ticker in symbol_set and earn_date:
                results.append({
                    "symbol": f"US.{ticker}",
                    "earnings_date": earn_date,
                    "source": "fmp",
                })
                _save_cached(f"US.{ticker}", earn_date, "fmp")

    except Exception as e:
        logger.debug(f"[Earnings] FMP error: {e}")

    return results


def get_upcoming_earnings(symbols: list[str],
                          days_ahead: int = 14,
                          fmp_api_key: str = "") -> list[dict]:
    """Get upcoming earnings dates from best available source.

    Returns list of {symbol, earnings_date, source} for symbols
    reporting within days_ahead.
    """
    results = []
    remaining = []

    # Check cache first
    for sym in symbols:
        cached = _get_cached(sym)
        if cached:
            try:
                earn_dt = datetime.strptime(cached, "%Y-%m-%d").date()
                if date.today() <= earn_dt <= date.today() + timedelta(days=days_ahead):
                    results.append({
                        "symbol": sym,
                        "earnings_date": cached,
                        "source": "cache",
                    })
                    continue
            except ValueError:
                pass
        remaining.append(sym)

    if not remaining:
        return results

    # Try FMP first (more reliable)
    if fmp_api_key:
        fmp_results = get_upcoming_earnings_fmp(
            remaining, fmp_api_key, days_ahead)
        if fmp_results:
            results.extend(fmp_results)
            found = set(r["symbol"] for r in fmp_results)
            remaining = [s for s in remaining if s not in found]

    # Fallback to Yahoo
    if remaining:
        yahoo_results = get_upcoming_earnings_yahoo(remaining, days_ahead)
        results.extend(yahoo_results)

    logger.info(f"[Earnings] 查询 {len(symbols)} 只, "
                f"发现 {len(results)} 只即将财报")
    return results
