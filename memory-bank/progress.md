# 项目进度

## 已完成 ✅

### 第一阶段：核心框架
- [x] 项目脚手架（config, requirements, .gitignore, README）
- [x] EventBus 事件总线
- [x] Order 数据模型
- [x] Logger 日志系统
- [x] YAML 配置加载工具
- [x] TechnicalIndicators 技术指标模块
- [x] Signal 模型 + BaseStrategy 抽象基类
- [x] RiskManager 风控引擎
- [x] PdtGuard PDT 规则守护
- [x] TelegramNotifier 通知模块
- [x] MarketData 行情数据接口
- [x] HistoryManager 历史数据缓存
- [x] PositionManager 仓位管理
- [x] Trader 交易执行
- [x] Backtester 回测引擎
- [x] BacktestReport 回测报告
- [x] TradingScheduler 交易时间调度
- [x] Engine 主引擎
- [x] main.py 入口
- [x] 三个内置策略（Momentum/MeanReversion/Breakout）
- [x] Memory Bank 初始化

### 测试覆盖
- [x] test_event_bus.py — 5 tests
- [x] test_order.py — 4 tests
- [x] test_indicators.py — 9 tests
- [x] test_risk_manager.py — 7 tests
- [x] test_pdt_guard.py — 6 tests
- [x] test_backtester.py — 2 tests

## 待开发 📋

### 第二阶段：策略增强
- [ ] 多因子选股模型
- [ ] 策略信号聚合器
- [ ] 期权交易支持
- [ ] 更多策略模板

### 第三阶段：AI/ML 增强
- [ ] 机器学习预测模型
- [ ] 自适应参数调优
- [ ] 智能仓位管理

### 基础设施改进
- [ ] SQLite 交易记录持久化
- [ ] Web 仪表盘
- [ ] 更完善的回测报告（图表）
- [ ] 策略参数优化工具

## 已知问题 ⚠️
- FutuOpenD 未安装时系统以 dry-run 模式运行
- Telegram 通知默认关闭，需配置 bot_token 和 chat_id
- 回测引擎暂不支持多标的同时回测
