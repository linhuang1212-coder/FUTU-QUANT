# 当前上下文

## 当前焦点
第一阶段核心框架已搭建完成，所有模块就位。

## 最近完成
- 项目脚手架搭建（config, requirements, README）
- EventBus 事件总线
- Order 模型 + Logger + Helpers
- TechnicalIndicators（MA/EMA/RSI/MACD/BB/ATR/VWAP/OBV）
- Signal 模型 + BaseStrategy 抽象基类
- RiskManager 风控引擎（仓位限制、日亏损、连续亏损冷却）
- PdtGuard PDT 规则守护
- TelegramNotifier 通知模块
- MarketData + HistoryManager 行情数据
- PositionManager + Trader 交易执行
- Backtester + BacktestReport 回测引擎
- TradingScheduler + Engine 主引擎
- 三个内置策略（Momentum/MeanReversion/Breakout）
- Memory Bank 初始化

## 下一步
- 安装依赖并验证系统能启动
- 配置 FutuOpenD 连接，测试模拟盘
- 用历史数据对三个内置策略进行回测
- 根据回测结果调优策略参数
- 配置 Telegram Bot 通知
- 模拟盘实盘跑一段时间验证稳定性
