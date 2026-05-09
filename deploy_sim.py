"""
部署 Strategy Lab Top3 到富途模拟盘 (SIMULATE)

使用 Ensemble_SharpeWeight 的信号，在富途模拟盘执行买入。
资金: 模拟盘默认 $100K (富途模拟盘初始资金)

Usage:
  python deploy_sim.py              # 预览信号 (dry-run)
  python deploy_sim.py --execute    # 执行下单
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.downloader import load_daily


def compute_ensemble_signals() -> dict[str, float]:
    """Compute Ensemble (Sharpe-weighted) allocation.

    Combines:
      - Trend_Vol_Target  (Sharpe 2.51)
      - Equal_Weight      (Sharpe 0.97)
      - Adaptive_AA       (Sharpe 0.95)
    """
    sharpes = {
        "trend": 2.51,
        "equal": 0.97,
        "aaa": 0.95,
    }
    total_s = sum(sharpes.values())

    # ── Trend Following + Vol Target ──
    trend_syms = ["SPY", "TLT", "GLD", "VEA", "DBC"]
    trend_weights = {}
    for sym in trend_syms:
        df = load_daily(sym)
        if df is None or len(df) < 252:
            continue
        closes = df["close"].values
        if closes[-1] <= np.mean(closes[-200:]):
            continue
        rets = np.diff(closes[-63:]) / closes[-63:-1]
        vol = float(np.std(rets) * np.sqrt(252))
        if vol > 0:
            trend_weights[sym] = 1.0 / vol
    if trend_weights:
        tw_total = sum(trend_weights.values())
        trend_weights = {s: v / tw_total for s, v in trend_weights.items()}
    else:
        trend_weights = {"SPY": 1.0}

    # ── Equal Weight + SMA200 ──
    eq_syms = ["SPY", "QQQ", "TLT", "GLD", "VEA", "EEM",
               "XLE", "XLU", "IEF", "SLV", "IWM", "XLK"]
    eligible = []
    for sym in eq_syms:
        df = load_daily(sym)
        if df is None or len(df) < 252:
            continue
        closes = df["close"].values
        if closes[-1] > np.mean(closes[-200:]):
            eligible.append(sym)
    if not eligible:
        eligible = ["SPY"]
    eq_weights = {s: 1.0 / len(eligible) for s in eligible}

    # ── Adaptive AA ──
    aaa_syms = ["SPY", "QQQ", "TLT", "GLD", "VEA", "EEM",
                "XLE", "XLU", "IEF", "DBC", "SLV", "IWM"]
    scores = {}
    vols = {}
    for sym in aaa_syms:
        df = load_daily(sym)
        if df is None or len(df) < 252:
            continue
        closes = df["close"].values
        past_126 = closes[-126] if len(closes) >= 126 else closes[0]
        mom = (closes[-1] / past_126) - 1 if past_126 > 0 else 0
        rets = np.diff(closes[-63:]) / closes[-63:-1]
        vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 10 else 999
        if vol <= 0:
            continue
        scores[sym] = mom / max(vol, 0.01)
        vols[sym] = vol
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    selected = [s for s, _ in ranked]
    inv_vols = {s: 1.0 / vols[s] for s in selected if s in vols}
    iv_total = sum(inv_vols.values()) if inv_vols else 1
    aaa_weights = {s: v / iv_total for s, v in inv_vols.items()}

    # ── Combine with Sharpe weights ──
    ensemble = {}
    for w_dict, strat_key in [(trend_weights, "trend"),
                               (eq_weights, "equal"),
                               (aaa_weights, "aaa")]:
        s_weight = sharpes[strat_key] / total_s
        for sym, w in w_dict.items():
            ensemble[sym] = ensemble.get(sym, 0) + w * s_weight

    return ensemble


def get_sim_account_info():
    """查询模拟盘账户信息。"""
    from futu import OpenSecTradeContext, OpenQuoteContext, RET_OK, TrdEnv

    trade_ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)
    ret, data = trade_ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
    if ret != RET_OK or data is None or data.empty:
        trade_ctx.close()
        return None
    row = data.iloc[0]
    info = {
        "total_assets": float(row.get("total_assets", 0)),
        "cash": float(row.get("cash", 0)),
        "market_val": float(row.get("market_val", 0)),
    }

    ret2, pos = trade_ctx.position_list_query(trd_env=TrdEnv.SIMULATE)
    positions = []
    if ret2 == RET_OK and pos is not None and not pos.empty:
        for _, r in pos.iterrows():
            qty = int(r.get("qty", 0))
            if qty > 0:
                positions.append({
                    "code": str(r.get("code", "")),
                    "qty": qty,
                    "cost": float(r.get("cost_price", 0)),
                    "market_val": float(r.get("market_val", 0)),
                })
    info["positions"] = positions
    trade_ctx.close()
    return info


def execute_orders(weights: dict[str, float], capital: float,
                   dry_run: bool = True):
    """Execute buy orders on Futu SIMULATE."""
    from futu import (OpenSecTradeContext, OpenQuoteContext, RET_OK,
                      TrdEnv, TrdSide, OrderType)

    trade_ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)
    quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

    # 1. 查询现有持仓
    ret, pos_data = trade_ctx.position_list_query(trd_env=TrdEnv.SIMULATE)
    existing = {}
    if ret == RET_OK and pos_data is not None and not pos_data.empty:
        for _, r in pos_data.iterrows():
            code = str(r.get("code", ""))
            qty = int(r.get("qty", 0))
            if qty > 0:
                existing[code] = qty

    target_codes = {f"US.{s}" if not s.startswith("US.") else s: w
                    for s, w in weights.items()}

    # 2. 卖出不在目标中的持仓
    for code, qty in existing.items():
        if code not in target_codes:
            time.sleep(1)
            ret, snap = quote_ctx.get_market_snapshot([code])
            if ret == RET_OK and snap is not None and not snap.empty:
                price = float(snap.iloc[0].get("last_price", 0))
            else:
                price = 0
            if price > 0 and qty > 0:
                print(f"  SELL  {code:10s}  qty={qty}  @${price:.2f}")
                if not dry_run:
                    time.sleep(1)
                    trade_ctx.place_order(
                        price=price, qty=qty, code=code,
                        trd_side=TrdSide.SELL, order_type=OrderType.NORMAL,
                        trd_env=TrdEnv.SIMULATE,
                    )

    # 3. 买入目标持仓
    time.sleep(2)
    ret, acct = trade_ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
    if ret == RET_OK and acct is not None and not acct.empty:
        available = float(acct.iloc[0].get("cash", capital))
    else:
        available = capital

    if dry_run:
        available = capital

    print(f"\n  Available cash: ${available:,.0f}\n")

    orders = []
    for code, w in sorted(target_codes.items(), key=lambda x: -x[1]):
        if w < 0.01:
            continue
        time.sleep(1)
        ret, snap = quote_ctx.get_market_snapshot([code])
        if ret != RET_OK or snap is None or snap.empty:
            print(f"  SKIP  {code} — no price data")
            continue
        price = float(snap.iloc[0].get("last_price", 0))
        if price <= 0:
            continue

        alloc = available * w
        qty = int(alloc / price)
        if qty < 1:
            print(f"  SKIP  {code:10s}  weight={w:.1%}  alloc=${alloc:.0f}  "
                  f"@${price:.2f} (qty < 1)")
            continue

        cost = qty * price
        print(f"  BUY   {code:10s}  weight={w:.1%}  qty={qty}  "
              f"@${price:.2f}  cost=${cost:,.0f}")
        orders.append({"code": code, "qty": qty, "price": price})

        if not dry_run:
            time.sleep(1)
            ret, data = trade_ctx.place_order(
                price=price, qty=qty, code=code,
                trd_side=TrdSide.BUY, order_type=OrderType.NORMAL,
                trd_env=TrdEnv.SIMULATE,
            )
            if ret == RET_OK:
                oid = str(data.iloc[0].get("order_id", ""))
                print(f"         -> order_id={oid}")
            else:
                print(f"         -> FAILED: {data}")

    quote_ctx.close()
    trade_ctx.close()
    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Execute orders (default: dry-run preview)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Deploy Strategy Lab -> Futu SIMULATE")
    print("=" * 60)

    # 计算信号
    weights = compute_ensemble_signals()
    print("\n  Ensemble (Sharpe-weighted) Target Allocation:")
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {sym:6s}  {w:.1%}")

    # 查询账户
    try:
        info = get_sim_account_info()
        if info:
            print(f"\n  Account: total=${info['total_assets']:,.0f}  "
                  f"cash=${info['cash']:,.0f}  "
                  f"positions={len(info['positions'])}")
            capital = info["cash"]
        else:
            print("\n  Cannot query account, using default capital")
            capital = 100_000
    except Exception as e:
        print(f"\n  Futu not connected: {e}")
        print("  Using dry-run mode with $100,000 capital")
        capital = 100_000
        args.execute = False

    # 下单
    mode = "LIVE EXECUTE" if args.execute else "DRY RUN"
    print(f"\n  Mode: {mode}")
    print(f"  Capital: ${capital:,.0f}")
    print(f"{'─' * 60}")

    execute_orders(weights, capital, dry_run=not args.execute)

    if not args.execute:
        print(f"\n  (Dry run — run with --execute to place orders)")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
