# 期权策略研究报告

## Futu OpenAPI 期权能力

### 支持的功能
- **期权链查询** (`get_option_chain`): 获取标的物的所有期权合约
  - 筛选：看涨/看跌、价内/价外、到期日范围
  - 数据：隐含波动率、Greeks (Delta/Gamma/Vega/Theta/Rho)、持仓量、成交量
- **期权报价** (`get_option_snapshot`): 实时期权价格
- **期权下单**: 支持美股期权交易（需要开通权限）
- **模拟盘**: 支持期权模拟交易

### 账户要求
- 需要开通美股期权交易权限
- 需要 Universal Account（证券综合账户）
- 最低保证金要求取决于策略类型

## $3,000 账户可行的期权策略

### 策略 1: 买入看涨期权替代持股（Long Call Replacement）

**逻辑**：当强信号触发时，用深度价内看涨期权替代直接买 TQQQ
- 优势：最大亏损 = 期权权利金（已知有限），杠杆更高
- 劣势：时间损耗、流动性可能不足
- 适用信号：Afternoon_Ext / Rebalance_2pm 高置信度信号

**实现可行性**：
- TQQQ 周期权流动性较好
- Delta > 0.8 的深度价内期权，效果类似持股但风险有限
- 每张期权控制 100 股，$3,000 大约能买 1-2 张
- **风险**：TQQQ 期权 bid-ask spread 较宽，日内策略频繁交易成本高

**建议**：仅用于 swing 信号特别强的时候，不适合日内高频使用

### 策略 2: Covered Call 增厚收益

**逻辑**：持有 TQQQ 时卖出虚值看涨期权（OTM Call）
- 优势：收取权利金，降低持有成本
- 劣势：限制上涨空间

**$3,000 账户限制**：
- 需要先持有至少 100 股 TQQQ（约 $5,000-6,000）
- **当前资金不够做标准 Covered Call**
- 替代：可以做 Poor Man's Covered Call（用 LEAPS 替代持股）

**建议**：资金不足，暂不可行

### 策略 3: Protective Put 止损保险

**逻辑**：持有 TQQQ 时买入虚值看跌期权作为保险
- 优势：限制下行风险
- 劣势：权利金成本
- $3,000 账户：每月花 $20-50 买保护 = 月成本 0.7-1.7%

**建议**：可行但成本较高，不建议在小账户上常规使用

## 结论与建议

| 策略 | 可行性 | 优先级 |
|------|--------|--------|
| Long Call 替代 | ✅ 可行（仅高置信信号） | 中 |
| Covered Call | ❌ 资金不足 | 低 |
| Poor Man's CC | ⚠️ 理论可行但复杂 | 低 |
| Protective Put | ✅ 可行但成本高 | 低 |

**当前阶段建议**：
1. 暂不将期权纳入自动化系统
2. 等账户增长到 $10,000+ 后再考虑 Covered Call
3. 如果要用期权，仅在 swing 策略产生高置信度信号时手动买入深度价内 Call
4. 专注优化已验证的日内策略（Rebalance_2pm / Afternoon_Ext / Vol_Squeeze）

## 技术实现备忘

```python
# Futu API 获取期权链示例
from futu import OpenQuoteContext, RET_OK

ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
ret, data = ctx.get_option_chain(
    code='US.TQQQ',
    start='2026-04-25',
    end='2026-05-02',
    option_type='CALL',
    option_cond_type='OUT_OF_THE_MONEY'
)
if ret == RET_OK:
    print(data)
ctx.close()
```
