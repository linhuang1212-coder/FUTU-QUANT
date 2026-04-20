"""Download extended 5min data from Futu API in monthly chunks.

Futu API allows ~80 days per query, so we fetch month by month
and merge into a single CSV.
"""
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from futu import OpenQuoteContext, RET_OK, KLType

HOST = "127.0.0.1"
PORT = 11111
SYMBOLS = ["US.TQQQ", "US.SOXL"]
DATA_DIR = Path("data_store/market_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 4, 18)


def download_5min(symbol: str):
    ticker = symbol.split(".")[-1]
    csv_path = DATA_DIR / f"{ticker}_5min.csv"

    ctx = OpenQuoteContext(host=HOST, port=PORT)
    all_dfs = []

    current = START_DATE
    chunk = 0
    while current < END_DATE:
        chunk_end = min(current + timedelta(days=28), END_DATE)
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        print(f"  {symbol} {start_str} ~ {end_str}...", end=" ", flush=True)

        page_key = None
        pages = []
        while True:
            kwargs = dict(code=symbol, start=start_str, end=end_str,
                          ktype=KLType.K_5M, max_count=1000)
            if page_key is not None:
                kwargs["page_req_key"] = page_key

            ret, data, page_key = ctx.request_history_kline(**kwargs)
            if ret == RET_OK and data is not None and len(data) > 0:
                pages.append(data)
            else:
                break
            if page_key is None:
                break
            time.sleep(0.5)

        if pages:
            chunk_df = pd.concat(pages, ignore_index=True)
            all_dfs.append(chunk_df)
            print(f"{len(chunk_df)} bars")
        else:
            print("0 bars")

        current = chunk_end + timedelta(days=1)
        chunk += 1
        time.sleep(1)

    ctx.close()

    if not all_dfs:
        print(f"  {symbol}: no data collected")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["time_key"], keep="last")
    df = df.sort_values("time_key").reset_index(drop=True)

    out = df[["time_key", "open", "high", "low", "close", "volume"]].copy()
    out.to_csv(csv_path, index=False)

    n_days = len(set(pd.to_datetime(out["time_key"]).dt.date))
    print(f"  {symbol}: {len(out)} total bars, {n_days} trading days -> {csv_path.name}")
    print(f"    Range: {out['time_key'].iloc[0]} ~ {out['time_key'].iloc[-1]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading 5min data from Futu API")
    print(f"Range: {START_DATE.date()} ~ {END_DATE.date()}")
    print("=" * 60)

    for sym in SYMBOLS:
        print(f"\n[{sym}]")
        download_5min(sym)

    print("\nDone!")
