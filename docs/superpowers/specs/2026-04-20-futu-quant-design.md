# FUTU-QUANT 量化交易系统设计文档

## 项目概述

基于富途证券 OpenAPI 的美股全自动量化交易系统。采用模块化架构，支持可插拔策略、完整风控体系和上下文记忆持久化。

## 核心约束

| 项目 | 决定 |
|------|------|
| 标的 | ETF 为主（杠杆 ETF 优先）+ 精选个股，强信号时可做期权 |
| 交易频率 | 日内 + 短线波段（不做中长线） |
| 自动化 | 全自动交易 |
| 运行环境 | 本地 Windows，FutuOpenD 网关 |
| 策略方法 | 分阶段：技术指标+量价分析 → 多因子 → AI/ML |
| 单笔风控 | 3-5% 资金 |
| 通知 | Telegram Bot |
| 本金 | $3,000 |
| 语言 | Python |

## 项目结构

```
FUTU-QUANT/
├── config/
│   ├── settings.yaml          # 全局配置（富途连接、资金、风控参数）
│   ├── strategies.yaml        # 策略配置（启用哪些策略、参数）
│   └── symbols.yaml           # 标的池配置（ETF/个股/期权列表）
│
├── core/
│   ├── engine.py              # 主引擎：调度器，协调所有模块
│   ├── scheduler.py           # 定时任务调度（开盘/收盘/盘中循环）
│   └── event_bus.py           # 轻量事件总线，模块间通信
│
├── data/
│   ├── market_data.py         # 行情数据获取（通过 FutuOpenD）
│   ├── history.py             # 历史数据管理与缓存
│   └── indicators.py          # 技术指标计算（MA/MACD/RSI/BOLL等）
│
├── strategy/
│   ├── base.py                # 策略基类（所有策略继承此类）
│   ├── momentum.py            # 动量策略
│   ├── mean_reversion.py      # 均值回归策略
│   └── breakout.py            # 突破策略
│
├── execution/
│   ├── trader.py              # 交易执行（下单/撤单/改单）
│   ├── position.py            # 仓位管理
│   └── order.py               # 订单数据模型
│
├── risk/
│   ├── risk_manager.py        # 风控引擎（止损/止盈/仓位限制）
│   └── pdt_guard.py           # PDT 规则守护（日内交易次数限制）
│
├── notification/
│   └── telegram_bot.py        # Telegram 通知推送
│
├── backtest/
│   ├── backtester.py          # 回测引擎
│   └── report.py              # 回测报告生成
│
├── utils/
│   ├── logger.py              # 日志系统
│   └── helpers.py             # 通用工具函数
│
├── memory-bank/
│   ├── project-brief.md       # 项目定义
│   ├── active-context.md      # 当前焦点
│   ├── system-patterns.md     # 架构模式
│   ├── tech-context.md        # 技术栈
│   ├── progress.md            # 进度追踪
│   └── strategy-journal.md    # 策略日志
│
├── data_store/                # 运行时数据存储（自动生成，gitignore）
│   ├── logs/
│   ├── trades/
│   └── cache/
│
├── main.py                    # 程序入口
├── requirements.txt           # 依赖管理
└── README.md                  # 项目说明
```

## 模块设计

### 1. 核心引擎（core/）

**engine.py — 主引擎**

系统入口，负责：
- 启动时连接 FutuOpenD 网关
- 初始化所有模块（数据、策略、风控、执行、通知）
- 运行主事件循环：行情接收 → 策略计算 → 风控审核 → 交易执行
- 优雅退出（断开连接、保存状态）

**scheduler.py — 定时调度**

基于美股交易时间（美东 9:30-16:00）：
- 盘前：加载配置、检查账户状态、推送启动通知
- 盘中：按设定间隔运行策略循环（1分钟/5分钟级别）
- 收盘前 15 分钟：强制平仓日内持仓
- 盘后：生成当日交易总结、推送 Telegram 日报

**event_bus.py — 事件总线**

轻量级发布-订阅模式，事件类型：
- `MARKET_DATA`：新行情到达
- `SIGNAL`：策略产生交易信号
- `ORDER`：订单创建/成交/取消
- `RISK_ALERT`：风控警告
- `SYSTEM`：系统状态变化

### 2. 数据模块（data/）

**market_data.py — 行情数据**

通过 futu-api 连接 FutuOpenD：
- 实时行情订阅（逐笔/K线）
- 按策略需求提供不同粒度数据（1分钟/5分钟/15分钟/日线）
- 连接断开自动重连

**history.py — 历史数据**

- 从富途拉取历史 K 线数据
- 本地 CSV/SQLite 缓存，避免重复请求
- 提供回测引擎所需的历史数据接口

**indicators.py — 技术指标**

基于 pandas/ta-lib 计算：
- 均线系列：MA、EMA、SMA
- 震荡指标：RSI、MACD、KDJ
- 通道指标：布林带、ATR
- 量价指标：OBV、VWAP
- 返回统一格式的 DataFrame，供策略模块使用

### 3. 策略模块（strategy/）

**base.py — 策略基类**

```python
class BaseStrategy:
    def on_bar(self, symbol, bar_data) -> Signal | None
    def on_tick(self, symbol, tick_data) -> Signal | None
    def get_params(self) -> dict
    def set_params(self, params: dict)
```

Signal 数据结构：
- direction: BUY / SELL
- strength: 0-100（信号强度）
- symbol: 标的代码
- suggested_type: STOCK / OPTION（强信号时建议期权）
- reason: 信号产生原因描述

**策略信号聚合规则：**
- 同方向信号叠加 → 提升信号强度，可加大仓位
- 矛盾信号 → 取消交易
- 信号强度 > 60 触发普通交易，> 80 可触发期权交易

**内置策略（第一阶段）：**

| 策略 | 核心逻辑 | 适用标的 | 持仓周期 |
|------|---------|---------|---------|
| 动量策略 | 短期均线上穿长期均线 + RSI 确认 + 量能放大 | TQQQ、SOXL 等杠杆 ETF | 1-5 天 |
| 均值回归 | 价格偏离布林带 + RSI 超买超卖 + 量能萎缩 | SPY、QQQ 等稳定 ETF | 日内-3 天 |
| 突破策略 | 突破关键位 + 量能急剧放大 + MACD 确认 | 个股和杠杆 ETF | 日内-数天 |

策略参数通过 `config/strategies.yaml` 配置，无需改代码。

### 4. 交易执行模块（execution/）

**trader.py — 交易执行**

通过 futu-api 下单：
- 支持市价单、限价单
- 下单前调用 RiskManager 审核
- 订单状态追踪（已提交/部分成交/全部成交/已取消）
- 下单失败自动重试（最多 3 次）

**position.py — 仓位管理**

- 实时同步富途账户持仓
- 计算每只标的的持仓比例
- 区分日内仓位和隔夜仓位

**order.py — 订单模型**

订单数据结构：symbol、direction、quantity、price、order_type、status、timestamp、strategy_name

### 5. 风控模块（risk/）

**risk_manager.py — 风控引擎**

下单前检查：
| 检查项 | 规则 | 不通过则 |
|--------|------|----------|
| 单笔亏损限制 | 最大亏损 ≤ 总资金 3-5%（可配置） | 拒绝下单 |
| 最大持仓比例 | 单只标的不超过总资金 40% | 缩减仓位 |
| 当日最大亏损 | 当天累计亏损 ≤ 总资金 8% | 暂停当日交易 |
| 连续亏损保护 | 连续亏损 N 次 → 冷却期 | 暂停交易 |
| 总仓位控制 | 总持仓不超过总资金 80% | 拒绝新开仓 |

持仓中监控：
| 监控项 | 动作 |
|--------|------|
| 移动止损 | 盈利达到阈值后启动，回撤触发止盈 |
| 硬止损 | 亏损达到阈值立即市价平仓 |
| 时间止损 | 日内仓位收盘前 15 分钟强制平仓 |

所有阈值通过 `config/settings.yaml` 配置。

**pdt_guard.py — PDT 规则守护**

SEC 规定账户 < $25,000 时，5 个交易日内最多 3 次日内交易：
- 实时追踪 5 天滚动窗口内的日内交易次数
- 已用 0-1 次 → 正常交易
- 已用 2 次 → Telegram 警告
- 已用 3 次 → 锁定日内交易，只允许波段策略
- 策略产生日内信号时自动判断额度

### 6. 通知模块（notification/）

**telegram_bot.py — Telegram 推送**

推送事件：
| 事件 | 内容 |
|------|------|
| 系统启动 | 连接状态、账户余额 |
| 开仓 | 标的、数量、价格、策略名、信号强度 |
| 平仓 | 标的、数量、价格、盈亏金额和百分比 |
| 止损触发 | 止损原因、亏损金额 |
| PDT 警告 | 剩余日内交易额度 |
| 当日总结 | 交易笔数、总盈亏、账户余额 |
| 系统异常 | 错误信息、重连状态 |

### 7. 回测模块（backtest/）

**backtester.py — 回测引擎**

- 加载历史数据，模拟策略运行
- 模拟撮合（考虑滑点和手续费）
- 支持多策略同时回测对比
- 与实盘策略共用同一套策略代码（BaseStrategy 接口统一）

**report.py — 回测报告**

生成指标：总收益率、年化收益、最大回撤、夏普比率、胜率、盈亏比

### 8. Memory Bank（memory-bank/）

项目上下文记忆持久化系统，用结构化 Markdown 文件记录项目全貌：

| 文件 | 内容 | 更新频率 |
|------|------|----------|
| `project-brief.md` | 项目目标、资金规模、风控原则、标的范围 | 很少变动 |
| `active-context.md` | 当前工作焦点、最近改动、下一步计划 | 每次会话更新 |
| `system-patterns.md` | 架构设计、模块关系、关键决策及理由 | 架构变动时更新 |
| `tech-context.md` | Python版本、依赖版本、FutuOpenD配置、API限制 | 环境变动时更新 |
| `progress.md` | 已完成模块、待开发功能、已知问题 | 每次会话更新 |
| `strategy-journal.md` | 回测结果、参数调优记录、实盘表现、经验教训 | 每次策略迭代更新 |

特性：
- 纯 Markdown 文件，git 版本化管理
- 每次 AI 会话开始时读取，快速恢复项目上下文
- strategy-journal.md 是量化项目特有的知识积累

## 数据流

```
市场行情 → data/market_data → core/engine → strategy/* → risk/risk_manager → execution/trader
                                                                                     │
                                                                              notification/telegram
```

1. Engine 启动后连接 FutuOpenD，订阅行情
2. MarketData 接收实时行情，Indicators 计算技术指标
3. Engine 将行情数据分发给所有启用的 Strategy
4. 策略产生交易信号（买/卖/持有）
5. RiskManager 审核信号（止损、仓位限制、PDT 规则）
6. 通过审核后 Trader 执行下单
7. Telegram 推送交易结果通知

## 配置管理

所有参数集中在 `config/` 目录的 YAML 文件中：

```yaml
# settings.yaml 示例
futu:
  host: "127.0.0.1"
  port: 11111
  trade_env: SIMULATE    # SIMULATE=模拟盘 / REAL=实盘

account:
  initial_capital: 3000
  currency: USD

risk:
  max_loss_per_trade_pct: 0.05
  max_daily_loss_pct: 0.08
  max_position_pct: 0.40
  max_total_position_pct: 0.80
  cooldown_after_consecutive_losses: 3

telegram:
  bot_token: "your-bot-token"
  chat_id: "your-chat-id"
```

关键原则：
- 模拟盘优先，改一行配置切实盘
- 策略可插拔，strategies.yaml 启用/禁用
- 所有参数可配置，不改代码即可调参
- 日志完整，每笔交易和风控动作都有记录

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 富途 API | futu-api |
| 数据处理 | pandas, numpy |
| 技术指标 | ta (技术分析库) |
| 配置管理 | PyYAML |
| 通知 | python-telegram-bot |
| 日志 | Python logging |
| 定时调度 | APScheduler |
| 数据存储 | SQLite (交易记录), CSV (行情缓存) |

## 分阶段实施

**第一阶段：核心框架 + 基础策略**
- 项目骨架搭建
- 连接 FutuOpenD 获取行情
- 实现 BaseStrategy + 1 个内置策略
- 风控框架
- Telegram 通知
- 回测引擎
- Memory Bank 初始化
- 模拟盘验证

**第二阶段：策略增强**
- 更多内置策略
- 多因子模型
- 策略信号聚合
- 期权交易支持

**第三阶段：AI/ML 增强**
- 机器学习预测模型
- 自适应参数调优
- 更智能的仓位管理
