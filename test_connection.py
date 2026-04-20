"""
FUTU-QUANT 连接测试脚本
测试 FutuOpenD 网关连接是否正常
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from futu import OpenQuoteContext, OpenSecTradeContext, RET_OK, TrdEnv


def test_quote_connection():
    print("=" * 60)
    print("  [1] 测试行情连接")
    print("=" * 60)

    try:
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        ret, data = ctx.get_market_snapshot(["US.AAPL"])
        if ret == RET_OK:
            print("[OK] 行情连接成功!")
            row = data.iloc[0]
            print(f"     AAPL 最新价: ${row['last_price']:.2f}")
        else:
            print(f"[FAIL] 行情请求失败: {data}")
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] 行情连接失败: {e}")
        print("       请确认 FutuOpenD 是否已启动并登录")
        return False


def test_trade_connection():
    print()
    print("=" * 60)
    print("  [2] 测试模拟盘交易连接")
    print("=" * 60)

    try:
        ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)
        ret, data = ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
        if ret == RET_OK:
            print("[OK] 模拟盘交易连接成功!")
            if not data.empty:
                print(f"     可用资金: ${data['cash'].iloc[0]:,.2f}")
                print(f"     总资产:   ${data['total_assets'].iloc[0]:,.2f}")
        else:
            print(f"[FAIL] 交易请求失败: {data}")
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] 交易连接失败: {e}")
        return False


def test_etf_quotes():
    print()
    print("=" * 60)
    print("  [3] 测试 ETF 标的池行情")
    print("=" * 60)

    symbols = ["US.TQQQ", "US.SQQQ", "US.SOXL", "US.SOXS", "US.TNA", "US.SPY", "US.QQQ", "US.IWM"]

    try:
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        ret, data = ctx.get_market_snapshot(symbols)
        if ret == RET_OK:
            print("[OK] ETF 行情获取成功!\n")
            print(f"     {'标的':<12} {'最新价':>10} {'涨跌额':>10} {'涨跌幅':>10}")
            print("     " + "-" * 46)
            for _, row in data.iterrows():
                code = row['code']
                last = row['last_price']
                # 通过开盘价计算涨跌
                open_price = row.get('open_price', last)
                change = last - open_price if open_price > 0 else 0
                change_pct = (change / open_price * 100) if open_price > 0 else 0
                print(f"     {code:<12} ${last:>8.2f}  {change:>+8.2f}  {change_pct:>+7.2f}%")
        else:
            print(f"[FAIL] 行情获取失败: {data}")
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] 连接失败: {e}")
        return False


def test_kline():
    print()
    print("=" * 60)
    print("  [4] 测试历史 K 线获取 (TQQQ 日线)")
    print("=" * 60)

    try:
        from futu import KLType, SubType
        ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

        ret_sub, _ = ctx.subscribe(["US.TQQQ"], [SubType.K_DAY])
        if ret_sub != RET_OK:
            print("[WARN] 订阅失败，尝试直接拉取...")

        ret, data = ctx.get_cur_kline("US.TQQQ", 10, KLType.K_DAY)
        if ret == RET_OK:
            print(f"[OK] 获取到 {len(data)} 条日线数据\n")
            print(f"     {'日期':<12} {'开盘':>8} {'最高':>8} {'最低':>8} {'收盘':>8} {'成交量':>12}")
            print("     " + "-" * 60)
            for _, row in data.tail(5).iterrows():
                date = str(row['time_key'])[:10]
                print(f"     {date:<12} {row['open']:>8.2f} {row['high']:>8.2f} {row['low']:>8.2f} {row['close']:>8.2f} {row['volume']:>12,.0f}")
        else:
            print(f"[FAIL] K 线获取失败: {data}")
        ctx.close()
        return True
    except Exception as e:
        print(f"[FAIL] K 线获取失败: {e}")
        return False


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  FUTU-QUANT 连接测试")
    print("=" * 60)
    print()

    q = test_quote_connection()
    t = test_trade_connection()
    e = test_etf_quotes()
    k = test_kline()

    print()
    print("=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    print(f"  行情连接:     {'OK' if q else 'FAIL'}")
    print(f"  模拟盘交易:   {'OK' if t else 'FAIL'}")
    print(f"  ETF 行情:     {'OK' if e else 'FAIL'}")
    print(f"  历史 K 线:    {'OK' if k else 'FAIL'}")
    print("=" * 60)

    if all([q, t, e, k]):
        print("\n  所有测试通过! 系统已就绪，可以开始量化交易。")
    else:
        print("\n  部分测试失败，请检查上方错误信息。")
    print()
