"""
Parquet 存储层 — 按年分区的行情 + 因子数据

文件布局:
  data_store/market_data/parquet/
    prices_2021.parquet
    prices_2022.parquet
    ...
    prices_2026.parquet

  data_store/factors/
    technical.parquet
    fundamental.parquet
    risk.parquet
    composite.parquet
    meta.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("factor_library.storage")

_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = _ROOT / "data_store" / "market_data" / "parquet"
FACTORS_DIR = _ROOT / "data_store" / "factors"


def _ensure_dirs():
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    FACTORS_DIR.mkdir(parents=True, exist_ok=True)


def save_prices(df: pd.DataFrame):
    """Save price data to year-partitioned Parquet files.

    df must have columns: date, symbol, open, high, low, close, volume
    """
    _ensure_dirs()
    if "date" not in df.columns:
        raise ValueError("DataFrame must have 'date' column")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    total = 0
    for year, group in df.groupby("year"):
        path = PARQUET_DIR / f"prices_{year}.parquet"
        chunk = group.drop(columns=["year"])

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, chunk], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
            combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        else:
            combined = chunk.sort_values(["symbol", "date"]).reset_index(drop=True)

        combined.to_parquet(path, index=False, engine="pyarrow")
        total += len(chunk)
        logger.info(f"[Storage] prices_{year}.parquet: {len(combined)} rows")

    logger.info(f"[Storage] 写入 {total} 行价格数据")


def load_prices(years: Optional[list[int]] = None,
                symbols: Optional[list[str]] = None) -> pd.DataFrame:
    """Load price data from Parquet files.

    Args:
        years: specific years to load (default: all)
        symbols: filter by symbols (default: all)
    """
    _ensure_dirs()
    files = sorted(PARQUET_DIR.glob("prices_*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        year = int(f.stem.split("_")[1])
        if years and year not in years:
            continue
        df = pd.read_parquet(f)
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    return result


def load_price_panel(symbols: Optional[list[str]] = None,
                     years: Optional[list[int]] = None) -> pd.DataFrame:
    """Load prices as a pivot table: index=date, columns=symbols, values=close."""
    df = load_prices(years=years, symbols=symbols)
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="date", columns="symbol", values="close")


def save_factors(df: pd.DataFrame, category: str):
    """Save factor values to Parquet.

    category: 'technical', 'fundamental', 'risk', 'composite'
    """
    _ensure_dirs()
    path = FACTORS_DIR / f"{category}.parquet"
    df.to_parquet(path, index=True, engine="pyarrow")
    logger.info(f"[Storage] {path.name}: {len(df)} rows x {len(df.columns)} factors")

    _update_meta(category, df)


def load_factors(category: str) -> pd.DataFrame:
    """Load factor values from Parquet."""
    path = FACTORS_DIR / f"{category}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _update_meta(category: str, df: pd.DataFrame):
    """Update meta.json with factor version info."""
    meta_path = FACTORS_DIR / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    meta[category] = {
        "factors": list(df.columns),
        "rows": len(df),
        "last_computed": datetime.now().isoformat(),
        "date_range": [str(df.index.min()), str(df.index.max())] if hasattr(df.index, 'min') else [],
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def snapshot_factors(category: str, version: str = "v1"):
    """Save a historical snapshot of factor data before formula changes."""
    src = FACTORS_DIR / f"{category}.parquet"
    if not src.exists():
        return
    hist_dir = FACTORS_DIR / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    dst = hist_dir / f"{category}_{version}_{date_str}.parquet"
    import shutil
    shutil.copy2(src, dst)
    logger.info(f"[Storage] 因子快照: {dst.name}")


def get_storage_stats() -> dict:
    """Report storage usage."""
    _ensure_dirs()
    stats = {"parquet_files": [], "factor_files": [], "total_mb": 0}

    for f in sorted(PARQUET_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1024 / 1024
        stats["parquet_files"].append({"name": f.name, "size_mb": round(size_mb, 2)})
        stats["total_mb"] += size_mb

    for f in sorted(FACTORS_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1024 / 1024
        stats["factor_files"].append({"name": f.name, "size_mb": round(size_mb, 2)})
        stats["total_mb"] += size_mb

    stats["total_mb"] = round(stats["total_mb"], 2)
    return stats
