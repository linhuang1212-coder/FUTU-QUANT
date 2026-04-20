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

### 第三阶段：多策略组合实盘系统
- [x] config/live.yaml 新增 portfolio 配置段（3 标的 x 11 个策略槽位）
- [x] run_live.py 重写为 MultiStrategyTrader（多标的多策略）
- [x] 指标预计算一次性添加所有 MA/EMA/RSI/BB/ATR/MACD
- [x] Sharpe 加权信号仲裁（score = sharpe_weight * strength）
- [x] 全仓轮流模式：同一时间只持有一个标的，$3,000 不拆分
- [x] 持仓优先检查卖出 → 空仓时选最强买入
- [x] hard stop 8% 止损保护
- [x] dry-run 测试通过：3 标的 11 策略全部正常加载和评估

### PDT 最大化：双层交易系统
- [x] PdtGuard 集成到 MultiStrategyTrader（5天3笔日内交易额度跟踪）
- [x] 新增盘中日内交易层（Layer 2）：5min K线 + RSI 超卖反弹 + BB 下轨
- [x] 日内入场条件：RSI(5) 从 <25 交叉上穿 + EMA(8)>EMA(20) 趋势确认
- [x] 日内出场条件：RSI>70 止盈 / +2% 目标利润 / -1.5% 止损
- [x] 收盘前 15 分钟强制平仓日内仓位（EOD force close）
- [x] PDT 余额检查：剩余 0 次时自动禁止日内交易
- [x] 双层互斥：swing 持仓时不做日内交易
- [x] 交易日志记录 PDT 剩余次数

### 策略验证套件（run_validation.py）
- [x] 日内 5min RSI 策略回测 -> **FAIL**（全部亏损，已禁用）
- [x] 滚动 Walk-Forward 验证（3yr+1yr x 8 窗口）-> 5/9 PASS, 3 WEAK, 1 FAIL
- [x] Monte Carlo 洗牌测试（1000 次）-> 交易边际存在（100th percentile）
- [x] 参数敏感性分析 -> 14/14 STABLE
- [x] 移除 momentum@TNA（WF FAIL），禁用日内交易层

### 新策略研究（第二轮）
- [x] 设计 4 个新策略：VWAP回归、多因子投票、波动率突破、快速均线交叉
- [x] 实现策略类：vwap_reversion.py, multi_factor.py, volatility_breakout.py, fast_ema_cross.py
- [x] 向量化信号生成器（run_new_strategy_scan.py）
- [x] Walk-Forward 参数扫描（1032 combos x 2 symbols）
- [x] 滚动 Walk-Forward 验证（3yr+1yr x 8 windows）
- [x] 结果：multi_factor 9 个 PASS，其余 3 个策略未通过
- [x] multi_factor 部署到 live.yaml（TQQQ OOS Sharpe 0.85 + SOXL OOS Sharpe 0.63）
- [x] 实盘策略总数：6 -> 8（新增 2 个 multi_factor）

### 日内策略 v2（顺势交易）
- [x] 分析 v1 RSI 逆势策略失败原因
- [x] 设计 3 个顺势日内策略：ORB、VWAP Trend、First Pullback
- [x] 获取 80 天 5min 数据（6204 bars/symbol）
- [x] 参数网格搜索（189 combos x 2 symbols）
- [x] Walk-Forward 验证：First Pullback@SOXL 10 个 PASS
- [x] 重写 run_live.py 日内层为 3 策略系统
- [x] 日内出场：VWAP 下破止损 / -3% 硬止损 / +4% 止盈 / EOD 强制平仓
- [x] 重新启用日内交易层
- [x] Dry-run 通过

### 市场状态过滤 + 动态仓位 + 新日内策略
- [x] VIX 过滤器：VIX>28 禁止入场，VIX>35 强制平仓（swing + intraday）
- [x] ADX 趋势确认：ADX>25 才允许趋势策略（momentum/breakout/ema_cross/vol_breakout）开仓
- [x] 动态仓位调整：EWMA 波动率目标 18%，VIX 分级缩仓，Drawdown Governor（-20% 减半）
- [x] 新模块 risk/vol_target.py：VolatilityTargetManager + MarketRegime 数据类
- [x] 隔夜跳空均值回归策略（gap_reversion）：开盘 gap-down 后等待反弹入场，30 分钟内退出
- [x] 杠杆 ETF 再平衡效应策略（rebalance_2pm）：大涨日 2pm 入场，利用 sponsor 再平衡买盘
- [x] run_intraday_scan.py 新增 2 个策略的回测函数 + 参数网格
- [x] live.yaml 新增 vol_target 配置段
- [x] run_intraday_scan.py 验证：gap_reversion（TQQQ Sharpe -3.07）和 rebalance_2pm（TQQQ Sharpe 0.29）未通过
- [x] 已从实盘 INTRADAY_CONFIGS 移除未验证策略，保留评估器代码备用
- [x] 日内策略仍为 4 个（已验证的 ORB + VWAP Trend + First Pullback）
- [x] 编译通过 + 45 tests 全部通过

### 测试覆盖 (45 tests)
- [x] test_event_bus.py — 5 tests
- [x] test_order.py — 4 tests
- [x] test_indicators.py — 9 tests
- [x] test_risk_manager.py — 7 tests
- [x] test_pdt_guard.py — 6 tests
- [x] test_backtester.py — 3 tests
- [x] test_optimizer.py — 3 tests
- [x] test_multi_timeframe.py — 8 tests

### 数据基础设施 + 10 年回测
- [x] data/downloader.py：Yahoo Finance 日线 + Alpha Vantage 5min + 统一加载接口
- [x] data/synthesizer.py：日线→5min 合成器（OHLC 约束 + U 型成交量 + 结构化路径）
- [x] 下载 10 年日线数据（TQQQ/SOXL 各 2511 bars，2016-2026）
- [x] run_intraday_scan.py + run_param_scan.py 集成本地数据优先加载
- [x] 10 年日内策略回测（2511 天 x 306 参数组合 x 2 标的）
- [x] 结果：ORB/First_Pullback/Gap_Reversion 全部失败
- [x] Rebalance_2pm@TQQQ 通过（Sharpe 2.47, WF Val Sharpe 2.73）
- [x] VWAP_Trend@TQQQ 弱通过（Sharpe 0.61, WF Val Sharpe 1.04）
- [x] 实盘日内配置更新为仅 Rebalance_2pm + VWAP_Trend（TQQQ only）
- [x] 编译通过 + 45 tests 全部通过

### 新策略研究 + 分段验证（AutoDL 服务器执行）
- [x] run_segmented_validation.py：10yr/5yr/3yr 分段 + COVID/加息/AI牛市压力测试 + 滚动 WF
- [x] 新增 4 个标的日线数据：QLD/SPXL/UPRO/TNA（各 2511 bars）
- [x] GitHub 仓库创建 + AutoDL 远程测试环境搭建
- [x] Swing 策略研究（S1~S5 + 原有策略）：**全部 FAIL**（最高 Sharpe 0.83）
- [x] 日内策略研究（I1~I3 + 已有策略）：
  - I2_Afternoon_Ext (1.5%/36bar): TQQQ Sharpe **3.89**, SOXL **4.28**, 全 6 标的 PASS
  - I3_Vol_Squeeze (0.5/24bar/1%): TQQQ Sharpe **9.77**, SOXL **9.84**（⚠️ 合成数据可能高估）
  - Rebalance_2pm (1.0%/48bar): TQQQ Sharpe **3.25**, QLD **3.61**, 5 标的 PASS
- [x] 通过策略部署到 run_live.py：TQQQ 3 个日内 + SOXL 2 个日内
- [x] 期权策略研究报告：$3,000 账户仅 Long Call 替代可行，暂不自动化

## 待开发 📋

### 策略深化
- [ ] 真实 5min 数据验证 Vol_Squeeze（需 Alpha Vantage API key）
- [ ] 期权交易支持（账户增长到 $10,000+ 后考虑）
- [ ] 资金拆分模式（split mode）支持同时持仓多个标的

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
