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

### 第二阶段：策略研究和回测增强
- [x] 回测报告增强：Sharpe/Sortino/Calmar/CAGR/月度收益/Buy&Hold 对比
- [x] 策略信号优化：宽容窗口、OR逻辑、分级信号强度
- [x] 新增 RsiReversalStrategy（RSI 反转策略，适合杠杆 ETF）
- [x] 参数扫描引擎：网格搜索 + Walk-Forward 验证 + 多标的扫描
- [x] 回测引擎升级：做空、信号强度仓位、O(n) 性能、ATR 止损
- [x] 多时间框架支持：K线重采样 + MultiTimeframeStrategy HTF 趋势过滤
- [x] 模拟盘验证管线：run_param_scan.py + run_simulation.py

### 测试覆盖 (45 tests)
- [x] test_event_bus.py — 5 tests
- [x] test_order.py — 4 tests
- [x] test_indicators.py — 9 tests
- [x] test_risk_manager.py — 7 tests
- [x] test_pdt_guard.py — 6 tests
- [x] test_backtester.py — 3 tests
- [x] test_optimizer.py — 3 tests
- [x] test_multi_timeframe.py — 8 tests

## 待开发 📋

### 第三阶段：策略深化
- [ ] 多因子选股模型
- [ ] 策略信号聚合器（多策略投票机制）
- [ ] 期权交易支持
- [ ] 更多策略模板（如 VWAP 策略、配对交易）

### 第四阶段：AI/ML 增强
- [ ] 机器学习预测模型
- [ ] 自适应参数调优（在线学习）
- [ ] 智能仓位管理（强化学习）

### 基础设施改进
- [ ] SQLite 交易记录持久化
- [ ] Web 仪表盘（可视化回测和实时监控）
- [ ] 回测图表输出（equity curve, drawdown chart）
- [ ] CI/CD 自动测试

## 已知问题 ⚠️
- FutuOpenD 未安装时系统以 dry-run 模式运行
- Telegram 通知默认关闭，需配置 bot_token 和 chat_id
- 模拟盘验证需要 FutuOpenD 持续运行在后台
