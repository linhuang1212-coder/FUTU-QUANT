"""
数据采集管道 — 批量下载行情 + 基本面

特性:
  - 分批下载 (每批 50 只, 间隔 2 秒, 防限速)
  - 断点续传 (记录进度, 中断后可续接)
  - 指数退避重试 (失败后 2s -> 4s -> 8s)
  - 拆股自动验证 (yfinance auto_adjust=True)
  - 基本面: SEC EDGAR frames API (免费, 无需 API key)
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from utils.logger import setup_logger

logger = setup_logger("factor_library.pipeline")

_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = _ROOT / "data_store" / "universe" / "download_progress.json"
BATCH_SIZE = 50
BATCH_DELAY = 2.0
MAX_RETRIES = 3

SEC_HEADERS = {
    "User-Agent": "FUTU-QUANT-Research admin@futuquant.local",
    "Accept-Encoding": "gzip, deflate",
}
SEC_BASE = "https://data.sec.gov"
SEC_RATE_LIMIT = 0.12  # 10 req/s max → ~8 req/s safe


# ──────────────────────────────────────────────────────
# CIK ↔ Ticker 映射
# ──────────────────────────────────────────────────────

_CIK_CACHE_PATH = _ROOT / "data_store" / "universe" / "cik_ticker_map.json"


def _load_cik_map() -> dict[str, int]:
    """加载 ticker -> CIK 映射."""
    if _CIK_CACHE_PATH.exists():
        try:
            return json.loads(_CIK_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cik_map(mapping: dict):
    _CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CIK_CACHE_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def refresh_cik_map() -> dict[str, int]:
    """从 SEC 下载 company_tickers.json 构建 ticker -> CIK 映射."""
    url = "https://www.sec.gov/files/company_tickers.json"
    logger.info("[Pipeline] 下载 SEC ticker-CIK 映射...")
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        mapping = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = entry.get("cik_str", 0)
            if ticker and cik:
                mapping[ticker] = int(cik)
        logger.info(f"[Pipeline] CIK 映射: {len(mapping)} 家公司")
        _save_cik_map(mapping)
        return mapping
    except Exception as e:
        logger.error(f"[Pipeline] CIK 映射下载失败: {e}")
        return _load_cik_map()


# ──────────────────────────────────────────────────────
# 行情数据 (yfinance)
# ──────────────────────────────────────────────────────

def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": [], "failed": [], "started_at": "", "batch_idx": 0}


def _save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def download_prices_batch(symbols: list[str], years: int = 5,
                          resume: bool = True) -> pd.DataFrame:
    """批量下载行情数据, 支持断点续传."""
    progress = _load_progress() if resume else {
        "completed": [], "failed": [], "started_at": "", "batch_idx": 0}

    if not progress["started_at"]:
        progress["started_at"] = datetime.now().isoformat()

    completed_set = set(progress["completed"])
    remaining = [s for s in symbols if s not in completed_set]

    if not remaining:
        logger.info("[Pipeline] 所有标的已下载完毕, 无需重复下载")
        return pd.DataFrame()

    logger.info(f"[Pipeline] 待下载: {len(remaining)} 只 "
                f"(已完成: {len(completed_set)}, 总计: {len(symbols)})")

    all_dfs = []
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        logger.info(f"[Pipeline] 批次 {batch_num}/{total_batches}: "
                    f"{batch[0]}...{batch[-1]} ({len(batch)} 只)")

        df = _download_batch_with_retry(batch, years)

        if df is not None and not df.empty:
            all_dfs.append(df)
            new_symbols = df["symbol"].unique().tolist()
            progress["completed"].extend(new_symbols)
            progress["failed"] = [s for s in batch if s not in new_symbols]
            logger.info(f"  成功: {len(new_symbols)} 只, "
                        f"失败: {len(batch) - len(new_symbols)} 只")
        else:
            progress["failed"].extend(batch)
            logger.warning(f"  批次 {batch_num} 全部失败")

        progress["batch_idx"] = batch_num
        _save_progress(progress)

        if batch_idx + BATCH_SIZE < len(remaining):
            time.sleep(BATCH_DELAY)

    progress["finished_at"] = datetime.now().isoformat()
    _save_progress(progress)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"[Pipeline] 下载完成: {result['symbol'].nunique()} 只, "
                    f"{len(result):,} 行")
        return result
    return pd.DataFrame()


def _download_batch_with_retry(symbols: list[str], years: int,
                               max_retries: int = MAX_RETRIES) -> Optional[pd.DataFrame]:
    """带指数退避重试的批量下载."""
    tickers_str = " ".join(symbols)
    end = datetime.now()
    start = datetime(end.year - years, end.month, end.day)

    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers_str,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                threads=True,
                progress=False,
                auto_adjust=True,
            )

            if data is None or data.empty:
                raise ValueError("Empty response")

            return _reshape_yf_data(data, symbols)

        except Exception as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"  下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"  等待 {wait} 秒后重试...")
                time.sleep(wait)

    return None


def _reshape_yf_data(data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """将 yfinance 多 ticker 宽格式转为长格式."""
    rows = []

    if isinstance(data.columns, pd.MultiIndex):
        available_symbols = data.columns.get_level_values(1).unique().tolist()
        for sym in available_symbols:
            try:
                sub = data.xs(sym, level=1, axis=1).copy()
                sub = sub.reset_index()
                date_col = [c for c in sub.columns if "date" in c.lower() or "Date" in str(c)]
                if date_col:
                    sub = sub.rename(columns={date_col[0]: "date"})
                elif "index" in sub.columns:
                    sub = sub.rename(columns={"index": "date"})
                else:
                    sub["date"] = sub.index

                col_map = {}
                for c in sub.columns:
                    cl = str(c).lower()
                    if cl in ("open", "high", "low", "close", "volume", "date"):
                        col_map[c] = cl

                sub = sub.rename(columns=col_map)
                required = {"date", "open", "high", "low", "close", "volume"}
                if not required.issubset(set(sub.columns)):
                    continue

                sub = sub[list(required)].copy()
                sub["symbol"] = sym
                sub["date"] = pd.to_datetime(sub["date"])
                sub = sub.dropna(subset=["close"])
                rows.append(sub)
            except Exception:
                continue
    else:
        if len(symbols) == 1:
            sub = data.reset_index()
            date_col = [c for c in sub.columns if "date" in c.lower() or "Date" in str(c)]
            if date_col:
                sub = sub.rename(columns={date_col[0]: "date"})

            col_map = {}
            for c in sub.columns:
                cl = str(c).lower()
                if cl in ("open", "high", "low", "close", "volume", "date"):
                    col_map[c] = cl
            sub = sub.rename(columns=col_map)

            required = {"date", "open", "high", "low", "close", "volume"}
            if required.issubset(set(sub.columns)):
                sub = sub[list(required)].copy()
                sub["symbol"] = symbols[0]
                sub["date"] = pd.to_datetime(sub["date"])
                sub = sub.dropna(subset=["close"])
                rows.append(sub)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result[["date", "symbol", "open", "high", "low", "close", "volume"]]


# ──────────────────────────────────────────────────────
# 基本面数据 (SEC EDGAR frames API — 完全免费)
# ──────────────────────────────────────────────────────

# 需要采集的 XBRL 财务概念 (us-gaap taxonomy)
SEC_CONCEPTS = {
    "Revenues":                        ("revenue",        "USD"),
    "NetIncomeLoss":                   ("net_income",     "USD"),
    "Assets":                          ("total_assets",   "USD"),
    "StockholdersEquity":              ("equity",         "USD"),
    "Liabilities":                     ("total_liabilities", "USD"),
    "OperatingIncomeLoss":             ("operating_income", "USD"),
    "EarningsPerShareBasic":           ("eps",            "USD/shares"),
    "CommonStockSharesOutstanding":    ("shares_out",     "shares"),
    "CashAndCashEquivalentsAtCarryingValue": ("cash",     "USD"),
    "LongTermDebt":                    ("long_term_debt", "USD"),
    "GrossProfit":                     ("gross_profit",   "USD"),
    "ResearchAndDevelopmentExpense":   ("rnd_expense",    "USD"),
}


def _sec_get(url: str) -> Optional[dict]:
    """SEC API 请求, 带重试."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"  SEC 429 限速, 等待 {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                logger.warning(f"  SEC 请求失败: {url}: {e}")
    return None


def _get_recent_period() -> str:
    """返回最近完成的财报季 (CY格式)."""
    now = datetime.now()
    year = now.year
    month = now.month
    # 延迟一个季度，确保数据已发布
    if month <= 3:
        return f"CY{year - 1}Q3"
    elif month <= 6:
        return f"CY{year - 1}Q4"
    elif month <= 9:
        return f"CY{year}Q1"
    else:
        return f"CY{year}Q2"


def download_fundamentals_sec(symbols: list[str],
                              cik_map: Optional[dict] = None) -> list[dict]:
    """用 SEC EDGAR frames API 批量下载基本面数据.

    frames API 每次返回整个市场某个指标的数据 → 只需 ~24 次 API 调用就能覆盖所有概念。
    """
    if cik_map is None:
        cik_map = refresh_cik_map()

    # 反向映射: CIK -> ticker
    cik_to_ticker = {v: k for k, v in cik_map.items()}
    symbol_set = set(s.upper() for s in symbols)

    period = _get_recent_period()
    prev_year_period = period.replace(str(int(period[2:6])), str(int(period[2:6]) - 1))
    logger.info(f"[Pipeline] SEC EDGAR 基本面: {len(symbols)} 只, "
                f"期间={period}, 概念={len(SEC_CONCEPTS)} 个")

    # 采集每个概念的全市场数据
    concept_data: dict[str, dict[int, float]] = {}
    total_concepts = len(SEC_CONCEPTS) * 2  # current + previous year for growth
    done_concepts = 0

    for concept, (field_name, unit) in SEC_CONCEPTS.items():
        unit_slug = unit.replace("/", "-per-")

        for p in [period, prev_year_period]:
            # 先尝试季度 (duration), 若失败则尝试瞬时值 (I)
            data = None
            for suffix in ["", "I"]:
                url = f"{SEC_BASE}/api/xbrl/frames/us-gaap/{concept}/{unit_slug}/{p}{suffix}.json"
                data = _sec_get(url)
                if data and data.get("data"):
                    break
                time.sleep(SEC_RATE_LIMIT)

            key = f"{field_name}_{p}"
            concept_data[key] = {}

            if data and data.get("data"):
                for entry in data["data"]:
                    cik = entry.get("cik", 0)
                    val = entry.get("val", 0)
                    if cik and val is not None:
                        concept_data[key][cik] = val

            done_concepts += 1
            if done_concepts % 4 == 0:
                logger.info(f"  SEC 进度: {done_concepts}/{total_concepts} 概念")
            time.sleep(SEC_RATE_LIMIT)

    # 组装每只股票的基本面
    results = []
    matched = 0

    for ticker in symbols:
        ticker_upper = ticker.upper()
        cik = cik_map.get(ticker_upper)
        if not cik:
            continue

        revenue = concept_data.get(f"revenue_{period}", {}).get(cik)
        revenue_prev = concept_data.get(f"revenue_{prev_year_period}", {}).get(cik)
        net_income = concept_data.get(f"net_income_{period}", {}).get(cik)
        assets = concept_data.get(f"total_assets_{period}", {}).get(cik)
        equity = concept_data.get(f"equity_{period}", {}).get(cik)
        liabilities = concept_data.get(f"total_liabilities_{period}", {}).get(cik)
        eps = concept_data.get(f"eps_{period}", {}).get(cik)
        shares = concept_data.get(f"shares_out_{period}", {}).get(cik)
        gross_profit = concept_data.get(f"gross_profit_{period}", {}).get(cik)
        long_term_debt = concept_data.get(f"long_term_debt_{period}", {}).get(cik)

        # 至少有收入或资产数据才算有效
        if not revenue and not assets:
            continue

        # 计算衍生指标
        roe = (net_income / equity) if (net_income and equity and equity > 0) else 0
        revenue_growth = (
            (revenue - revenue_prev) / abs(revenue_prev)
            if (revenue and revenue_prev and revenue_prev != 0) else 0
        )
        gross_margin = (
            gross_profit / revenue if (gross_profit and revenue and revenue > 0) else 0
        )
        debt_equity = (
            (liabilities or long_term_debt or 0) / equity
            if (equity and equity > 0) else 0
        )
        pe = 0  # 需要价格数据后计算

        rec = {
            "symbol": ticker_upper,
            "sector": "",
            "industry": "",
            "market_cap": 0,
            "pe": pe,
            "pb": 0,
            "dividend_yield": 0,
            "revenue_growth": round(revenue_growth, 4),
            "gross_margin": round(gross_margin, 4),
            "debt_equity": round(debt_equity, 4),
            "avg_volume": 0,
            "roe": round(roe, 4),
            "revenue": revenue or 0,
            "net_income": net_income or 0,
            "total_assets": assets or 0,
            "equity": equity or 0,
            "eps": eps or 0,
            "shares_outstanding": shares or 0,
        }
        results.append(rec)
        matched += 1

    logger.info(f"[Pipeline] SEC 基本面完成: 有效={matched}, "
                f"无CIK映射={len(symbols) - len(cik_map)}")
    return results


def download_fundamentals(symbols: list[str],
                          max_workers: int = 10) -> list[dict]:
    """下载基本面数据 (SEC EDGAR frames API).

    已从 yfinance 迁移到 SEC EDGAR, 完全免费, 无限速问题.
    """
    return download_fundamentals_sec(symbols)


# ──────────────────────────────────────────────────────
# 补充: 行业分类 (SEC submissions API)
# ──────────────────────────────────────────────────────

def download_sector_info(symbols: list[str],
                         cik_map: Optional[dict] = None,
                         max_workers: int = 5) -> list[dict]:
    """从 SEC submissions API 获取行业分类 (SIC code)."""
    if cik_map is None:
        cik_map = refresh_cik_map()

    # SIC 大类映射
    sic_sectors = {
        range(100, 1000): "Agriculture",
        range(1000, 1500): "Mining",
        range(1500, 1800): "Construction",
        range(2000, 4000): "Manufacturing",
        range(4000, 5000): "Transportation",
        range(5000, 5200): "Wholesale",
        range(5200, 6000): "Retail",
        range(6000, 6800): "Finance",
        range(7000, 9000): "Services",
        range(9100, 9730): "Government",
    }

    def _sic_to_sector(sic: int) -> str:
        for r, name in sic_sectors.items():
            if sic in r:
                return name
        return "Other"

    results = []

    def _fetch_one(ticker: str) -> Optional[dict]:
        cik = cik_map.get(ticker.upper())
        if not cik:
            return None
        cik_padded = str(cik).zfill(10)
        url = f"{SEC_BASE}/submissions/CIK{cik_padded}.json"
        data = _sec_get(url)
        if not data:
            return None
        sic = int(data.get("sic", "0") or "0")
        return {
            "symbol": ticker.upper(),
            "sector": _sic_to_sector(sic),
            "industry": data.get("sicDescription", ""),
            "sic_code": sic,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in symbols}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception:
                pass
            if done % 200 == 0:
                logger.info(f"  行业分类进度: {done}/{len(symbols)}")
            time.sleep(SEC_RATE_LIMIT)

    logger.info(f"[Pipeline] 行业分类完成: {len(results)} 只")
    return results


def incremental_update(symbols: list[str]) -> pd.DataFrame:
    """增量更新: 只下载最新数据 (最近 5 天)."""
    logger.info(f"[Pipeline] 增量更新 {len(symbols)} 只 (最近5天)")
    tickers_str = " ".join(symbols)

    try:
        data = yf.download(
            tickers_str,
            period="5d",
            threads=True,
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        return _reshape_yf_data(data, symbols)
    except Exception as e:
        logger.error(f"[Pipeline] 增量更新失败: {e}")
        return pd.DataFrame()
