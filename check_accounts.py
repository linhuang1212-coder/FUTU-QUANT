"""查看所有账户信息"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from futu import OpenSecTradeContext, TrdEnv, TrdMarket, RET_OK


def check_account(env_name, trd_env):
    print(f"\n{'=' * 55}")
    print(f"  {env_name}")
    print(f"{'=' * 55}")

    try:
        ctx = OpenSecTradeContext(host="127.0.0.1", port=11111)

        # 账户资金
        ret, data = ctx.accinfo_query(trd_env=trd_env)
        if ret == RET_OK and not data.empty:
            for _, row in data.iterrows():
                print(f"  总资产:       ${row.get('total_assets', 0):>12,.2f}")
                print(f"  现金:         ${row.get('cash', 0):>12,.2f}")
                print(f"  持仓市值:     ${row.get('market_val', 0):>12,.2f}")
                print(f"  可用资金:     ${row.get('avl_withdrawal_cash', 0):>12,.2f}")
                print(f"  账户货币:     {row.get('currency', 'N/A')}")
        else:
            print(f"  获取资金信息失败: {data}")

        # 持仓
        ret2, pos_data = ctx.position_list_query(trd_env=trd_env)
        if ret2 == RET_OK and not pos_data.empty:
            print(f"\n  当前持仓:")
            print(f"  {'标的':<12} {'数量':>8} {'成本价':>10} {'市价':>10} {'市值':>12} {'盈亏':>12} {'盈亏%':>8}")
            print(f"  {'-' * 74}")
            for _, row in pos_data.iterrows():
                print(f"  {row['code']:<12} {row['qty']:>8.0f} {row.get('cost_price', 0):>10.2f} {row.get('market_val', 0) / row['qty'] if row['qty'] > 0 else 0:>10.2f} ${row.get('market_val', 0):>10,.2f} ${row.get('pl_val', 0):>10,.2f} {row.get('pl_ratio', 0) * 100:>+7.2f}%")
        elif ret2 == RET_OK:
            print(f"\n  当前无持仓")
        else:
            print(f"\n  获取持仓失败: {pos_data}")

        ctx.close()
    except Exception as e:
        print(f"  查询失败: {e}")


if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  FUTU-QUANT 账户总览")
    print("=" * 55)

    check_account("模拟盘", TrdEnv.SIMULATE)
    check_account("实盘", TrdEnv.REAL)

    print(f"\n{'=' * 55}")
    print()
