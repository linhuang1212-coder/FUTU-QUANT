"""
碎股 API 测试脚本
在模拟盘测试 Moomoo OpenAPI 是否支持 qty < 1 的碎股下单。
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from futu import (
    OpenQuoteContext, OpenSecTradeContext, RET_OK,
    TrdEnv, TrdSide, OrderType, SubType,
)


HOST = "127.0.0.1"
PORT = 11111
TRADE_ENV = TrdEnv.REAL


def test_fractional_order():
    """Test placing a fractional share order (qty=0.1 SPY) on SIMULATE."""
    print("=" * 60)
    print("  碎股 API 测试 (模拟盘)")
    print("=" * 60)

    quote_ctx = OpenQuoteContext(host=HOST, port=PORT)
    trade_ctx = OpenSecTradeContext(host=HOST, port=PORT)

    try:
        # Step 1: 获取 SPY 实时报价
        print("\n[1] 获取 SPY 报价...")
        ret, snap = quote_ctx.get_market_snapshot(["US.SPY"])
        if ret != RET_OK:
            print(f"  FAIL: {snap}")
            return False
        price = float(snap["last_price"].iloc[0])
        print(f"  SPY 最新价: ${price:.2f}")

        # Step 2: 模拟盘账户信息
        print("\n[2] 模拟盘账户...")
        ret, acc = trade_ctx.accinfo_query(trd_env=TRADE_ENV)
        if ret == RET_OK and not acc.empty:
            cash = float(acc["cash"].iloc[0])
            print(f"  可用资金: ${cash:,.2f}")
        else:
            print(f"  WARN: 无法获取账户信息: {acc}")

        # Step 3: 测试碎股下单 (qty=0.1)
        print("\n[3] 尝试碎股下单: SPY qty=0.1 限价单...")
        limit_price = round(price * 0.50, 2)  # 低于市价50%，绝不会成交
        print(f"  限价: ${limit_price:.2f} (低于市价，避免成交)")

        ret, data = trade_ctx.place_order(
            price=limit_price,
            qty=0.1,
            code="US.SPY",
            trd_side=TrdSide.BUY,
            order_type=OrderType.NORMAL,
            trd_env=TRADE_ENV,
        )

        if ret == RET_OK:
            order_id = data["order_id"].iloc[0]
            print(f"  SUCCESS! 碎股下单成功! Order ID: {order_id}")
            print(f"  qty=0.1, price=${limit_price:.2f}")

            # Step 4: 立即撤单
            print("\n[4] 撤销测试订单...")
            ret2, data2 = trade_ctx.modify_order(
                modify_order_op=2,  # CANCEL
                order_id=order_id,
                qty=0.1,
                price=limit_price,
                trd_env=TRADE_ENV,
            )
            if ret2 == RET_OK:
                print(f"  订单已撤销")
            else:
                print(f"  撤单失败: {data2}")
                print(f"  请手动在 Moomoo 客户端撤销 Order ID: {order_id}")

            return True
        else:
            print(f"  FAIL: 碎股下单失败")
            print(f"  错误信息: {data}")
            return False

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False
    finally:
        quote_ctx.close()
        trade_ctx.close()


def test_sgov_availability():
    """Check if SGOV is quotable (碎股标的可用性)."""
    print("\n" + "=" * 60)
    print("  SGOV / BIL / GLD / TLT 标的可用性测试")
    print("=" * 60)

    symbols = ["US.SGOV", "US.BIL", "US.GLD", "US.TLT", "US.VEA",
               "US.SPY", "US.QQQ", "US.IWM"]

    quote_ctx = OpenQuoteContext(host=HOST, port=PORT)
    try:
        ret, data = quote_ctx.get_market_snapshot(symbols)
        if ret == RET_OK:
            print(f"\n  {'标的':<12} {'最新价':>10} {'成交量':>14} {'状态':>6}")
            print("  " + "-" * 48)
            for _, row in data.iterrows():
                code = row["code"]
                last = row["last_price"]
                vol = row.get("volume", 0)
                status = "OK" if last > 0 else "N/A"
                print(f"  {code:<12} ${last:>8.2f} {vol:>12,.0f}   {status}")
        else:
            print(f"  FAIL: {data}")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        quote_ctx.close()


if __name__ == "__main__":
    print()
    ok = test_fractional_order()
    test_sgov_availability()

    print("\n" + "=" * 60)
    if ok:
        print("  结论: Moomoo OpenAPI 支持碎股下单!")
        print("  可以继续实现 Cash Parking 和动量轮动策略")
    else:
        print("  结论: Moomoo OpenAPI 不支持碎股下单")
        print("  需要考虑替代方案 (手动操作 / 仅整股 ETF)")
    print("=" * 60)
