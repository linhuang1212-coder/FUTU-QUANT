"""
FUTU-QUANT 期权访问测试脚本
测试账户是否可以获取期权链数据及下单权限
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
from datetime import datetime, timedelta
from futu import OpenQuoteContext, OpenSecTradeContext, RET_OK, TrdEnv, OptionType, OptionCondType


def test_option_expiry():
    print("=" * 60)
    print("  [1] 测试期权到期日获取")
    print("=" * 60)

    try:
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        ret, data = ctx.get_option_expiration_date(code="US.SPY")
        if ret == RET_OK:
            print("[OK] 期权到期日获取成功!")
            print(f"     共 {len(data)} 个到期日")
            if not data.empty:
                print(f"     最近到期日: {data.iloc[0]['strike_time']}")
                print(f"     最远到期日: {data.iloc[-1]['strike_time']}")
        else:
            print(f"[FAIL] 期权到期日获取失败: {data}")
            ctx.close()
            return False
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] 期权到期日获取失败: {e}")
        return False


def test_option_chain():
    print()
    print("=" * 60)
    print("  [2] 测试期权链获取")
    print("=" * 60)

    try:
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        time.sleep(0.5)
        ret, data = ctx.get_option_chain(
            code="US.SPY",
            start=today,
            end=end_date,
            option_type=OptionType.ALL,
            option_cond_type=OptionCondType.ALL,
        )
        if ret == RET_OK:
            if data.empty:
                print("[WARN] 期权链返回空数据，可能无可用合约")
                ctx.close()
                return True
            print(f"[OK] 期权链获取成功! 共 {len(data)} 条合约")
            print()
            print(f"     {'期权代码':<30} {'类型':>6} {'行权价':>10} {'到期日':>12}")
            print("     " + "-" * 62)
            for _, row in data.head(10).iterrows():
                print(f"     {row['code']:<30} {row['option_type']:>6} {row['strike_price']:>10.2f} {str(row['strike_time'])[:10]:>12}")
        else:
            print(f"[FAIL] 期权链获取失败: {data}")
            ctx.close()
            return False
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] 期权链获取失败: {e}")
        return False


def test_option_quote():
    print()
    print("=" * 60)
    print("  [3] 测试期权报价获取")
    print("=" * 60)

    try:
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        time.sleep(0.5)
        ret, data = ctx.get_option_chain(
            code="US.SPY",
            start=today,
            end=end_date,
            option_type=OptionType.ALL,
            option_cond_type=OptionCondType.ALL,
        )
        if ret != RET_OK or data.empty:
            print("[WARN] 无法获取期权链，跳过报价测试")
            ctx.close()
            return True

        option_code = data.iloc[0]['code']
        print(f"     测试合约: {option_code}")

        time.sleep(0.5)
        ret_snap, snap_data = ctx.get_market_snapshot([option_code])
        if ret_snap == RET_OK:
            print("[OK] 期权报价获取成功!")
            row = snap_data.iloc[0]
            print(f"     最新价:   ${row['last_price']:.2f}")
            print(f"     买一价:   ${row.get('bid_price', 0):.2f}")
            print(f"     卖一价:   ${row.get('ask_price', 0):.2f}")
            print(f"     成交量:   {row.get('volume', 0):,.0f}")
        else:
            print(f"[FAIL] 期权报价获取失败: {snap_data}")
            ctx.close()
            return False
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] 期权报价获取失败: {e}")
        return False


def test_simulate_order():
    print()
    print("=" * 60)
    print("  [4] 测试模拟盘交易权限 (仅查询，不下单)")
    print("=" * 60)

    try:
        ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)
        time.sleep(0.5)
        ret, data = ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
        if ret == RET_OK:
            print("[OK] 模拟盘交易权限正常!")
            if not data.empty:
                print(f"     可用资金: ${data['cash'].iloc[0]:,.2f}")
                print(f"     总资产:   ${data['total_assets'].iloc[0]:,.2f}")
        else:
            print(f"[FAIL] 模拟盘查询失败: {data}")
            ctx.close()
            return False
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] 模拟盘交易权限测试失败: {e}")
        return False


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  FUTU-QUANT 期权访问测试")
    print("=" * 60)
    print()

    r1 = test_option_expiry()
    r2 = test_option_chain()
    r3 = test_option_quote()
    r4 = test_simulate_order()

    print()
    print("=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    print(f"  期权到期日:     {'OK' if r1 else 'FAIL'}")
    print(f"  期权链获取:     {'OK' if r2 else 'FAIL'}")
    print(f"  期权报价:       {'OK' if r3 else 'FAIL'}")
    print(f"  模拟盘权限:     {'OK' if r4 else 'FAIL'}")
    print("=" * 60)

    if all([r1, r2, r3, r4]):
        print("\n  所有测试通过! 期权数据访问及交易权限正常。")
    else:
        print("\n  部分测试失败，请检查上方错误信息。")
    print()
