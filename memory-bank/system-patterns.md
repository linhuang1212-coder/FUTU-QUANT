# 系统架构模式

## 整体架构
模块化架构，通过轻量级事件总线通信：

```
市场行情 → data/market_data → core/engine → strategy/* → risk/risk_manager → execution/trader
                                                                                     │
                                                                              notification/telegram
```

## 模块职责

| 模块 | 职责 |
|------|------|
| core/engine | 主引擎，协调所有模块，运行主事件循环 |
| core/event_bus | 发布-订阅事件总线，模块间解耦通信 |
| core/scheduler | 美股交易时间管理，收盘前强制平仓 |
| data/market_data | 连接 FutuOpenD 获取实时行情 |
| data/history | 历史数据缓存和管理 |
| data/indicators | 技术指标计算（纯函数，无状态） |
| strategy/base | 策略接口定义（BaseStrategy + Signal） |
| strategy/* | 具体策略实现（可插拔） |
| execution/trader | 交易执行，集成风控和 PDT 检查 |
| execution/position | 仓位追踪和管理 |
| execution/order | 订单数据模型 |
| risk/risk_manager | 交易前风控检查和仓位计算 |
| risk/pdt_guard | PDT 日内交易规则守护 |
| notification/telegram | Telegram 消息推送 |
| backtest/* | 回测引擎和报告生成 |

## 关键设计决策

1. **事件总线而非直接调用**: 模块间通过 EventBus 通信，降低耦合度
2. **策略基类模式**: 所有策略继承 BaseStrategy，统一 on_bar/on_tick 接口
3. **风控前置**: 每笔交易必须通过 RiskManager 审核才能执行
4. **配置驱动**: 所有参数在 YAML 文件中，运行时不改代码
5. **回测复用**: 回测引擎和实盘共用同一套策略代码
6. **ZoneInfo 而非 pytz**: 使用 Python 内置时区支持
