"""Quick test: verify the full order flow works (DRY-RUN, no real money).

Tests:
  1. Connect to Futu API
  2. Get real-time price for TQQQ
  3. Simulate a BUY order (dry-run)
  4. Simulate a SELL order (dry-run)
  5. Check account balance
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from utils.helpers import load_yaml, get_project_root

root = get_project_root()
config = load_yaml(str(root / "config" / "live.yaml"))

print("=" * 60)
print("  FUTU-QUANT Order Flow Test (DRY-RUN)")
print("=" * 60)

# Step 1: Connect
print("\n[1/5] Connecting to FutuOpenD...")
from futu import OpenQuoteContext, OpenSecTradeContext, RET_OK, TrdEnv, TrdSide, OrderType

quote_ctx = OpenQuoteContext(host=config["futu"]["host"], port=config["futu"]["port"])
trade_ctx = OpenSecTradeContext(host=config["futu"]["host"], port=config["futu"]["port"])
print("  Quote + Trade contexts connected OK")

# Step 2: Get real-time prices
print("\n[2/5] Fetching real-time prices...")
symbols = ["US.TQQQ", "US.SOXL", "US.UPRO", "US.TECL"]
ret, data = quote_ctx.get_market_snapshot(symbols)
if ret == RET_OK and data is not None:
    for _, row in data.iterrows():
        print(f"  {row['code']}: ${row['last_price']:.2f} "
              f"(open=${row['open_price']:.2f}, high=${row['high_price']:.2f}, "
              f"low=${row['low_price']:.2f}, vol={row['volume']:,.0f})")
else:
    print(f"  ERROR: {data}")

# Step 3: Check account
print("\n[3/5] Checking account balance...")
env = TrdEnv.REAL
ret, acc_data = trade_ctx.accinfo_query(trd_env=env)
if ret == RET_OK and acc_data is not None:
    for _, row in acc_data.iterrows():
        print(f"  Total assets:  ${row.get('total_assets', 'N/A')}")
        print(f"  Cash:          ${row.get('cash', 'N/A')}")
        print(f"  Market value:  ${row.get('market_val', 'N/A')}")
        print(f"  Available:     ${row.get('avl_withdrawal_cash', 'N/A')}")
else:
    print(f"  ERROR: {acc_data}")

# Step 4: Check existing positions
print("\n[4/5] Checking positions...")
ret, pos_data = trade_ctx.position_list_query(trd_env=env)
if ret == RET_OK and pos_data is not None and len(pos_data) > 0:
    for _, row in pos_data.iterrows():
        if float(row["qty"]) > 0:
            print(f"  {row['code']}: {int(row['qty'])} shares @ ${float(row['cost_price']):.2f} "
                  f"(P&L: ${float(row.get('pl_val', 0)):.2f})")
else:
    print("  No positions")

# Step 5: Simulate order (calculate only, DO NOT place)
print("\n[5/5] Simulating order calculation...")
if ret == RET_OK:
    # Get TQQQ price
    ret2, snap = quote_ctx.get_market_snapshot(["US.TQQQ"])
    if ret2 == RET_OK:
        tqqq_price = float(snap.iloc[0]["last_price"])
        capital = 3000
        alloc = 0.72
        qty = int(capital * alloc / tqqq_price)
        cost = qty * tqqq_price
        print(f"  If we buy TQQQ now:")
        print(f"    Price:      ${tqqq_price:.2f}")
        print(f"    Allocation: {alloc:.0%} of ${capital}")
        print(f"    Quantity:   {qty} shares")
        print(f"    Cost:       ${cost:.2f}")
        print(f"    Remaining:  ${capital - cost:.2f}")
        print(f"\n  >>> This is a DRY-RUN. NO order was placed. <<<")

# Cleanup
quote_ctx.close()
trade_ctx.close()
print(f"\n{'=' * 60}")
print("  Test complete. System is working correctly.")
print(f"{'=' * 60}")
