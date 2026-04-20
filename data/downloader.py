"""Market data downloader: Yahoo Finance (daily) + Alpha Vantage (5min intraday).

Yahoo Finance:  unlimited daily history, free, no API key
Alpha Vantage:  2 years of 5min data in 24 monthly slices, free API key required

Data is stored as CSV in data_store/market_data/{symbol}_{interval}.csv
with columns: time_key, open, high, low, close, volume

Usage:
    python -m data.downloader                      # download all
    python -m data.downloader --symbols TQQQ SOXL  # specific symbols
    python -m data.downloader --daily-only          # skip 5min (no AV key)
    python -m data.downloader --av-key YOUR_KEY     # Alpha Vantage key
"""

import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data_store" / "market_data"
DEFAULT_SYMBOLS = ["TQQQ", "SOXL"]
AV_BASE_URL = "https://www.alphavantage.co/query"


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _csv_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol}_{interval}.csv"


def _normalize_symbol(sym: str) -> str:
    """Strip exchange prefix: 'US.TQQQ' -> 'TQQQ'."""
    return sym.split(".")[-1] if "." in sym else sym


# ── Yahoo Finance: daily data ───────────────────────────────────


def download_daily_yf(symbol: str, years: int = 10) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    df = ticker.history(start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        interval="1d", auto_adjust=True)
    if df.empty:
        print(f"  [WARN] No daily data for {symbol}")
        return pd.DataFrame()

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "time_key", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["time_key"] = pd.to_datetime(df["time_key"]).dt.strftime("%Y-%m-%d")
    df = df[["time_key", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("time_key").reset_index(drop=True)
    return df


def save_daily(symbol: str, years: int = 10) -> int:
    """Download and save daily data. Returns row count."""
    path = _csv_path(symbol, "daily")
    df = download_daily_yf(symbol, years)
    if df.empty:
        return 0
    df.to_csv(path, index=False)
    print(f"  {symbol} daily: {len(df)} bars -> {path.name}")
    return len(df)


# ── Alpha Vantage: 5-min intraday (2 years in 24 slices) ───────


def download_5min_av(symbol: str, av_key: str, months: int = 24) -> pd.DataFrame:
    """Download up to 2 years of 5min data from Alpha Vantage extended API."""
    all_dfs = []
    slices_done = 0

    for y in range(1, 3):  # year1, year2
        for m in range(1, 13):
            if slices_done >= months:
                break
            slice_name = f"year{y}month{m}"
            params = {
                "function": "TIME_SERIES_INTRADAY_EXTENDED",
                "symbol": symbol,
                "interval": "5min",
                "slice": slice_name,
                "apikey": av_key,
                "adjusted": "true",
            }
            print(f"    Fetching {symbol} 5min {slice_name}...", end=" ", flush=True)
            try:
                resp = requests.get(AV_BASE_URL, params=params, timeout=30)
                if resp.status_code != 200:
                    print(f"HTTP {resp.status_code}")
                    continue
                if "Error" in resp.text[:200] or "premium" in resp.text[:500].lower():
                    print(f"API error: {resp.text[:120]}")
                    break

                from io import StringIO
                chunk = pd.read_csv(StringIO(resp.text))
                if chunk.empty or len(chunk) < 10:
                    print("empty")
                    continue
                all_dfs.append(chunk)
                print(f"{len(chunk)} bars")
                slices_done += 1
            except Exception as e:
                print(f"error: {e}")
                continue

            # Rate limit: 5 calls/min for free tier
            time.sleep(13)
        if slices_done >= months:
            break

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.rename(columns={"time": "time_key"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df.sort_values("time_key").drop_duplicates(subset=["time_key"]).reset_index(drop=True)
    df["time_key"] = df["time_key"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df[["time_key", "open", "high", "low", "close", "volume"]]
    return df


def download_5min_yf(symbol: str) -> pd.DataFrame:
    """Fallback: download ~60 days of 5min data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="60d", interval="5m", auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        dt_col: "time_key", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["time_key"] = pd.to_datetime(df["time_key"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df[["time_key", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("time_key").reset_index(drop=True)
    return df


def save_5min(symbol: str, av_key: Optional[str] = None, months: int = 24) -> int:
    """Download and save 5min data. Returns row count."""
    path = _csv_path(symbol, "5min")

    if av_key:
        df = download_5min_av(symbol, av_key, months)
        if not df.empty:
            # Merge with existing data if present
            if path.exists():
                existing = pd.read_csv(path)
                df = pd.concat([existing, df], ignore_index=True)
                df["time_key"] = pd.to_datetime(df["time_key"])
                df = df.sort_values("time_key").drop_duplicates(
                    subset=["time_key"]).reset_index(drop=True)
                df["time_key"] = df["time_key"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df.to_csv(path, index=False)
            print(f"  {symbol} 5min: {len(df)} bars -> {path.name}")
            return len(df)

    print(f"  [INFO] No AV key, using Yahoo Finance (max ~60 days)")
    df = download_5min_yf(symbol)
    if df.empty:
        return 0
    df.to_csv(path, index=False)
    print(f"  {symbol} 5min: {len(df)} bars -> {path.name}")
    return len(df)


# ── Unified data loader ─────────────────────────────────────────


def load_local_data(symbol: str, interval: str = "daily") -> Optional[pd.DataFrame]:
    """Load data from local CSV. Returns None if not found.

    symbol: 'TQQQ' or 'US.TQQQ' (prefix stripped automatically)
    interval: 'daily' or '5min'
    """
    sym = _normalize_symbol(symbol)
    path = _csv_path(sym, interval)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def load_daily(symbol: str) -> Optional[pd.DataFrame]:
    return load_local_data(symbol, "daily")


def load_5min(symbol: str) -> Optional[pd.DataFrame]:
    return load_local_data(symbol, "5min")


def data_summary():
    """Print summary of all local data files."""
    _ensure_dir()
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        print("No local data files found.")
        return
    print(f"\nLocal data in {DATA_DIR}:")
    print(f"{'File':<30} {'Rows':>10} {'Start':>12} {'End':>12}")
    print("-" * 66)
    for f in files:
        try:
            df = pd.read_csv(f)
            start = df["time_key"].iloc[0][:10]
            end = df["time_key"].iloc[-1][:10]
            print(f"{f.name:<30} {len(df):>10,} {start:>12} {end:>12}")
        except Exception as e:
            print(f"{f.name:<30} ERROR: {e}")


# ── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--av-key", type=str, default=None,
                        help="Alpha Vantage API key (free at alphavantage.co)")
    parser.add_argument("--daily-only", action="store_true")
    parser.add_argument("--5min-only", action="store_true", dest="fivemin_only")
    parser.add_argument("--months", type=int, default=24,
                        help="Months of 5min data to download (max 24)")
    parser.add_argument("--years", type=int, default=10,
                        help="Years of daily data to download")
    parser.add_argument("--summary", action="store_true",
                        help="Show data summary and exit")
    args = parser.parse_args()

    _ensure_dir()

    if args.summary:
        data_summary()
        return

    print("=" * 60)
    print("FUTU-QUANT Data Downloader")
    print("=" * 60)

    for sym in args.symbols:
        sym = _normalize_symbol(sym)
        print(f"\n[{sym}]")

        if not args.fivemin_only:
            save_daily(sym, args.years)

        if not args.daily_only:
            save_5min(sym, args.av_key, args.months)

    print("\n" + "=" * 60)
    data_summary()


if __name__ == "__main__":
    main()
