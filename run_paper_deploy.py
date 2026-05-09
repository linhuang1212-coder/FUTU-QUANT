"""
Deploy Top 3 strategies to paper trading (SQLite simulation).

This script:
1. Loads current market data
2. Generates signals for each of the Top 3 strategies
3. Records target holdings in SQLite for daily NAV tracking
4. Outputs current allocation recommendations

Top 3 from Strategy Lab:
  1. Ensemble (Sharpe-weighted: Trend+EqualWeight+AAA)
  2. Trend Following + Vol Target
  3. Adaptive Asset Allocation

Usage:
  python run_paper_deploy.py              # show signals & targets
  python run_paper_deploy.py --init       # initialize paper portfolios
  python run_paper_deploy.py --update     # update daily NAV
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.downloader import load_daily

DB_PATH = Path(__file__).resolve().parent / "data_store" / "strategy_lab_paper.db"
CAPITAL = 10_000


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS portfolios (
            strategy TEXT PRIMARY KEY,
            capital  REAL NOT NULL,
            holdings TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nav_history (
            strategy TEXT NOT NULL,
            date     TEXT NOT NULL,
            nav      REAL NOT NULL,
            holdings TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (strategy, date)
        );
    """)
    return conn


def _get_prices(symbols: list[str]) -> dict[str, float]:
    """Get latest prices from cached data."""
    prices = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and not df.empty:
            prices[sym] = float(df["close"].iloc[-1])
    return prices


def compute_trend_vol_signals() -> dict[str, float]:
    """Trend Following + Vol Target: SMA200 filter + inverse-vol weights."""
    symbols = ["SPY", "TLT", "GLD", "VEA", "DBC"]
    target_vol = 0.10
    weights = {}

    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 252:
            continue
        closes = df["close"].values
        current = closes[-1]
        sma200 = np.mean(closes[-200:])

        if current <= sma200:
            continue

        rets = np.diff(closes[-63:]) / closes[-63:-1]
        vol = float(np.std(rets) * np.sqrt(252))
        if vol > 0:
            weights[sym] = 1.0 / vol

    if not weights:
        weights = {"SPY": 1.0}

    total = sum(weights.values())
    weights = {s: v / total for s, v in weights.items()}

    port_vol = sum((w * (1.0 / w * total)) ** 2 for s, w in weights.items())
    # Already normalized; just cap leverage
    for s in weights:
        weights[s] = min(weights[s], 0.4)

    return weights


def compute_aaa_signals() -> dict[str, float]:
    """Adaptive Asset Allocation: momentum/vol scoring, top 5."""
    symbols = ["SPY", "QQQ", "TLT", "GLD", "VEA", "EEM",
               "XLE", "XLU", "IEF", "DBC", "SLV", "IWM"]
    scores = {}
    vols = {}

    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 252:
            continue
        closes = df["close"].values
        current = closes[-1]
        past_126 = closes[-126] if len(closes) >= 126 else closes[0]
        mom = (current / past_126) - 1 if past_126 > 0 else 0

        rets = np.diff(closes[-63:]) / closes[-63:-1]
        vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 10 else 999
        if vol <= 0:
            continue

        scores[sym] = mom / max(vol, 0.01)
        vols[sym] = vol

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    selected = [s for s, _ in ranked]

    inv_vols = {s: 1.0 / vols[s] for s in selected if s in vols}
    total = sum(inv_vols.values()) if inv_vols else 1
    return {s: v / total for s, v in inv_vols.items()}


def compute_equal_weight_signals() -> dict[str, float]:
    """Equal weight with SMA200 momentum filter."""
    symbols = ["SPY", "QQQ", "TLT", "GLD", "VEA", "EEM",
               "XLE", "XLU", "IEF", "SLV", "IWM", "XLK"]
    eligible = []
    for sym in symbols:
        df = load_daily(sym)
        if df is None or len(df) < 252:
            continue
        closes = df["close"].values
        if closes[-1] > np.mean(closes[-200:]):
            eligible.append(sym)

    if not eligible:
        eligible = ["SPY"]
    return {s: 1.0 / len(eligible) for s in eligible}


def show_signals():
    """Display current signals for all three strategies."""
    print("=" * 60)
    print(f"  Strategy Lab — Paper Trading Signals")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Capital: ${CAPITAL:,}")
    print("=" * 60)

    strategies = {
        "Trend_Vol_Target": compute_trend_vol_signals(),
        "Adaptive_AA": compute_aaa_signals(),
        "Equal_Weight": compute_equal_weight_signals(),
    }

    all_syms = set()
    for w in strategies.values():
        all_syms.update(w.keys())
    prices = _get_prices(list(all_syms))

    for name, weights in strategies.items():
        print(f"\n--- {name} ---")
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            price = prices.get(sym, 0)
            alloc = w * CAPITAL
            qty = int(alloc / price) if price > 0 else 0
            print(f"  {sym:6s}  weight={w:.1%}  ${alloc:,.0f}  "
                  f"qty={qty}  @${price:.2f}")

    # Ensemble (Sharpe-weighted of above 3)
    # From lab: Trend=2.51, EqWeight=0.97, AAA=0.95
    sharpes = {"Trend_Vol_Target": 2.51, "Equal_Weight": 0.97, "Adaptive_AA": 0.95}
    total_s = sum(sharpes.values())
    print(f"\n--- Ensemble_SharpeWeight ---")
    ensemble_weights = {}
    for strat_name, strat_w in strategies.items():
        s_weight = sharpes[strat_name] / total_s
        for sym, w in strat_w.items():
            ensemble_weights[sym] = ensemble_weights.get(sym, 0) + w * s_weight

    for sym, w in sorted(ensemble_weights.items(), key=lambda x: -x[1]):
        price = prices.get(sym, 0)
        alloc = w * CAPITAL
        qty = int(alloc / price) if price > 0 else 0
        print(f"  {sym:6s}  weight={w:.1%}  ${alloc:,.0f}  "
              f"qty={qty}  @${price:.2f}")


def init_portfolios():
    """Initialize paper portfolios in SQLite."""
    conn = _connect()
    now = datetime.now().isoformat()

    strategies = {
        "Trend_Vol_Target": compute_trend_vol_signals(),
        "Adaptive_AA": compute_aaa_signals(),
        "Equal_Weight": compute_equal_weight_signals(),
    }

    for name, weights in strategies.items():
        conn.execute(
            "INSERT OR REPLACE INTO portfolios (strategy, capital, holdings, created_at) "
            "VALUES (?, ?, ?, ?)",
            (name, CAPITAL, json.dumps(weights), now)
        )
        conn.execute(
            "INSERT OR REPLACE INTO nav_history (strategy, date, nav, holdings) "
            "VALUES (?, ?, ?, ?)",
            (name, now[:10], CAPITAL, json.dumps(weights))
        )
    conn.commit()
    conn.close()
    print(f"  Initialized {len(strategies)} paper portfolios in {DB_PATH}")


def update_nav():
    """Update daily NAV for all paper portfolios."""
    conn = _connect()
    today = datetime.now().strftime("%Y-%m-%d")

    rows = conn.execute("SELECT strategy, capital, holdings FROM portfolios").fetchall()
    for row in rows:
        name = row["strategy"]
        capital = row["capital"]
        weights = json.loads(row["holdings"])

        all_syms = list(weights.keys())
        prices = _get_prices(all_syms)

        nav = 0
        for sym, w in weights.items():
            price = prices.get(sym, 0)
            alloc = w * capital
            qty = int(alloc / price) if price > 0 else 0
            nav += qty * price

        cash = capital - sum(
            int(w * capital / prices.get(s, 1)) * prices.get(s, 0)
            for s, w in weights.items()
        )
        nav += max(cash, 0)

        conn.execute(
            "INSERT OR REPLACE INTO nav_history (strategy, date, nav, holdings) "
            "VALUES (?, ?, ?, ?)",
            (name, today, nav, json.dumps(weights))
        )
        print(f"  {name}: NAV=${nav:,.0f}")

    conn.commit()
    conn.close()
    print(f"  NAV updated for {today}")


def main():
    parser = argparse.ArgumentParser(description="Paper trading deployment")
    parser.add_argument("--init", action="store_true", help="Initialize portfolios")
    parser.add_argument("--update", action="store_true", help="Update daily NAV")
    args = parser.parse_args()

    if args.init:
        init_portfolios()
    elif args.update:
        update_nav()
    else:
        show_signals()


if __name__ == "__main__":
    main()
