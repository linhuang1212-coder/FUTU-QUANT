# 美股因子库设计文档

> 生成时间: 2026-05-05 | 版本: v1.1 (自审优化版)

## 1. 目标

构建覆盖**全美股 (4,500+ 只)** 的因子库系统，支持：
- **选股**: 多因子打分排序，找被低估的优质股
- **择时**: 市场状态判断，优化开仓/平仓时机
- **风控**: 过滤危险标的，控制下行风险
- **因子研究**: 快速发现、验证和迭代新因子

## 2. 架构总览

```
┌─────────────────────────────────────────────────────┐
│                    用户/策略层                         │
│   选股引擎 | 择时信号 | 风控过滤 | 因子研究交互         │
├─────────────────────────────────────────────────────┤
│                   向量检索层 (ChromaDB)                │
│   因子特征向量 | 相似股票搜索 | 因子聚类 | 异常检测      │
├─────────────────────────────────────────────────────┤
│                   因子计算层                           │
│   技术因子 | 基本面因子 | 波动因子 | 风险因子 | 另类因子  │
├─────────────────────────────────────────────────────┤
│                   数据存储层                           │
│   Parquet (行情+因子值) | SQLite (元数据+基本面)        │
├─────────────────────────────────────────────────────┤
│                   数据采集层                           │
│   Yahoo Finance | SEC/EDGAR | Futu API | GitHub列表   │
└─────────────────────────────────────────────────────┘
```

## 3. 数据采集层

### 3.1 股票列表获取 (Ticker Universe)

| 数据源 | 内容 | 格式 | 免费 | 更新频率 |
|--------|------|------|------|---------|
| **rreichel3/US-Stock-Symbols** (GitHub) | NYSE + NASDAQ + AMEX 全量 ticker | JSON/TXT | ✅ | 每日自动 |
| **adanos-software/free-ticker-database** | 38,437 US stocks + 15,009 ETFs | CSV/Parquet | ✅ | 定期 |
| **NASDAQ Screener CSV** | 8,000+ tickers with market cap | CSV | ✅ | 实时 |

**推荐方案**: 从 GitHub repo 拉取全量 ticker 列表，按市值过滤活跃股票 (日均成交量 > 10万股)。

### 3.2 行情数据 (OHLCV)

| 数据源 | 方法 | 限制 | 速度 |
|--------|------|------|------|
| **Yahoo Finance (yfinance)** | `yf.download(tickers, threads=8)` | 无 API key，无限制 | ~200 只/分钟 |
| **Futu API** | `request_history_kline` | 需 FutuOpenD，有频率限制 | ~30 只/分钟 |

**推荐方案**: yfinance 为主，4500 只 ≈ 22 分钟可下载完毕。每日增量更新只需几分钟。

> **⚠️ 风险缓解**: yfinance 是非官方 API，可能被限速。措施：
> 1. 分批下载（每批 100 只，间隔 2 秒），内置指数退避重试
> 2. 断点续传：记录已下载的 ticker，中断后可续接
> 3. 备用源自动切换：yfinance → Futu API → 本地缓存

### 3.3 基本面数据 (Fundamentals)

| 数据源 | 免费额度 | 字段 | API Key |
|--------|---------|------|---------|
| **SecuritiesDB** | 100 req/min，无限 | PE, ROE, Revenue Growth, DCF | 不需要 |
| **Fundamentals API** (SEC数据) | 免费 5000+ 公司 | Income/Balance/Cash Flow | 需要(免费) |
| **FMP** | 250 req/day | 全面，但额度小 | 需要(免费) |
| **yfinance** | 无限 | PE, PB, Market Cap, Sector | 不需要 |

**推荐方案**: 
1. yfinance 获取 PE/PB/Market Cap/Sector (免费无限)
2. SecuritiesDB 补充 ROE/Revenue Growth (免费无 key)
3. FMP 作为 fallback (250/天)

> **⚠️ 性能注意** (v1.1): yfinance `info` 属性每只约 1-2 秒，4500 只串行需 75-150 分钟。
> 优化方案：使用 `concurrent.futures.ThreadPoolExecutor(max_workers=10)` 并行获取，
> 预计 10-15 分钟完成。按行业分批，每批间隔 5 秒防限速。

## 4. 数据存储层

### 4.1 Parquet (行情 + 因子值)

```
data_store/
├── universe/
│   ├── tickers.parquet          # 全量 ticker 元数据
│   └── active_universe.parquet   # 活跃标的列表 (过滤后)
├── market_data/
│   ├── prices.parquet            # 全量日线 (date, symbol, O, H, L, C, V)
│   └── prices_latest.parquet     # 最近 60 天 (快速加载)
├── factors/
│   ├── technical.parquet         # 技术因子 (date, symbol, MOM_1M, VOL_20D, ...)
│   ├── fundamental.parquet       # 基本面因子 (date, symbol, EP, ROE, ...)
│   ├── risk.parquet              # 风险因子 (date, symbol, BETA, DOWNVOL, ...)
│   └── composite.parquet         # 合成分数 (date, symbol, score)
└── cache/                        # 临时缓存 (CSV, 兼容旧代码)
```

**选择 Parquet 的理由**:
- 4500只 x 5年日线 ≈ 570万行 → Parquet 压缩后 ~50-80MB (CSV 需 ~500MB)
- 列式存储：读取单个因子列极快，不用加载全部数据
- Pandas 原生支持 `pd.read_parquet()` / `df.to_parquet()`

**按年分区策略** (v1.1 优化):
```
market_data/
├── prices_2021.parquet    # 历史数据 (只读)
├── prices_2022.parquet
├── prices_2023.parquet
├── prices_2024.parquet
├── prices_2025.parquet
└── prices_2026.parquet    # 当年数据 (每日追加)
```
增量更新只需读写当年文件 (~1/5 数据量)，历史文件不动。

### 4.2 数据质量保障 (v1.1)

#### 退市股处理
- 只关注当前活跃股票，退市股不保留
- 每月刷新 ticker 列表，自动剔除退市/不活跃标的
- 回测结论需注意存活偏差，在报告中标注

#### 拆股/复权处理 (v1.1 新增)
拆股（Stock Split）会导致历史价格不连续，产生虚假动量/波动信号。

**必须使用复权价格 (Adjusted Price)**：
- yfinance `auto_adjust=True`（默认）→ 返回的就是前复权价格 ✅
- 检查点：下载后验证是否有单日涨跌 >50% 的异常跳变
- 常见案例：NVDA (2024年6月 10:1拆股)、TSLA (2022年 3:1拆股)、AAPL (2020年 4:1拆股)

**验证逻辑**:
```python
def validate_no_splits(prices: pd.DataFrame, threshold: float = 0.5):
    """检测未复权的拆股痕迹"""
    returns = prices.pct_change().abs()
    suspicious = returns[returns > threshold].dropna(how='all')
    if not suspicious.empty:
        logger.warning(f"疑似未复权数据: {suspicious.index.tolist()}")
    return suspicious.empty
```

**Volume 复权**：拆股后成交量也需要调整（10:1 拆股后历史 volume x10），
yfinance 默认处理了，但需要在数据入库时验证。

### 4.3 SQLite (元数据 + 基本面)

```sql
-- 股票元数据
CREATE TABLE universe (
    symbol       TEXT PRIMARY KEY,
    name         TEXT,
    exchange     TEXT,    -- NYSE / NASDAQ / AMEX
    sector       TEXT,    -- Technology / Healthcare / ...
    industry     TEXT,
    market_cap   REAL,
    avg_volume   REAL,    -- 30日平均成交量
    is_active    INTEGER, -- 1=活跃, 0=退市/不活跃
    last_updated TEXT
);

-- 季度基本面数据
CREATE TABLE fundamentals (
    symbol       TEXT,
    report_date  TEXT,    -- 财报日期
    pe           REAL,
    pb           REAL,
    roe          REAL,
    revenue_growth REAL,
    gross_margin REAL,
    debt_equity  REAL,
    fcf_yield    REAL,
    dividend_yield REAL,
    piotroski_f  INTEGER, -- F-Score 0-9
    PRIMARY KEY (symbol, report_date)
);
```

## 5. 因子计算层

### 5.1 因子分类体系 (40+ 因子)

#### A. 技术因子 (已有 8 个 → 扩展到 15 个)

| 因子名 | 公式 | 用途 | 状态 |
|--------|------|------|------|
| MOM_1M | P/P(-21) - 1 | 短期动量 | ✅ 已有 |
| MOM_3M | P/P(-63) - 1 | 中期动量 | ✅ 已有 |
| MOM_6M | P/P(-126) - 1 | 中期动量 | ✅ 已有 |
| MOM_12M | P/P(-252) - 1 | 长期动量 | ✅ 已有 |
| MOM_12M_1M | MOM_12M - MOM_1M | 去噪动量 (Jegadeesh) | 🆕 |
| VOL_20D | std(ret, 20) * √252 | 短期波动 | ✅ 已有 |
| VOL_60D | std(ret, 60) * √252 | 中期波动 | ✅ 已有 |
| TURNOVER | avg_vol(20) / avg_vol(100) | 相对换手率 | ✅ 已有 |
| REVERSAL | -sum(ret, 5) | 短期反转 | ✅ 已有 |
| RSI_14 | RSI(14) | 超买超卖 | 🆕 |
| MACD_HIST | MACD histogram | 趋势强度 | 🆕 |
| BB_WIDTH | (upper - lower) / middle | 波动率扩张 | 🆕 |
| ATR_PCT | ATR(14) / Close | 标准化波动 | 🆕 |
| PRICE_SMA200 | Close / SMA(200) - 1 | 趋势偏离 | 🆕 |
| VOLUME_SURGE | Vol / avg_vol(20) | 异常放量 | 🆕 |

#### B. 基本面因子 (已有 3 个 → 扩展到 12 个)

| 因子名 | 公式 | 用途 | 状态 |
|--------|------|------|------|
| EP | 1/PE | 盈利收益率 | ✅ 已有 |
| ROE | Net Income / Equity | 盈利质量 | ✅ 已有 |
| REV_GROWTH | Revenue YoY% | 增长 | ✅ 已有 |
| BP | 1/PB | 账面价值比 | 🆕 |
| FCF_YIELD | FCF / Market Cap | 自由现金流 | 🆕 |
| GROSS_MARGIN | Gross Profit / Revenue | 毛利率 | 🆕 |
| DEBT_EQUITY | Total Debt / Equity | 杠杆率 | 🆕 |
| DIV_YIELD | Dividend / Price | 股息率 | 🆕 |
| PIOTROSKI_F | F-Score (0-9) | 财务健康综合 | 🆕 |
| EARNINGS_SURPRISE | Actual EPS / Est EPS - 1 | 盈余超预期 | 🆕 |
| ACCRUALS | (NI - CFO) / TA | 应计异常 | 🆕 |
| CAPEX_RATIO | CapEx / Revenue | 资本密集度 | 🆕 |

#### C. 风险因子 (新增 8 个)

| 因子名 | 公式 | 用途 |
|--------|------|------|
| BETA | β vs SPY (252D) | 系统性风险 |
| DOWNVOL | std(neg_ret, 60) * √252 | 下行波动 |
| MAX_DD_60D | 60天最大回撤 | 尾部风险 |
| SKEWNESS | skew(ret, 60) | 收益分布偏斜 |
| KURTOSIS | kurt(ret, 60) | 肥尾程度 |
| SORTINO_RATIO | ret / downvol | 风险调整收益 |
| CALMAR_RATIO | CAGR / MaxDD | 回撤调整收益 |
| CORR_SPY | corr(ret, SPY_ret, 60) | 市场相关性 |

#### D. 波动/期权因子 (已有 2 个 → 扩展到 5 个)

| 因子名 | 公式 | 用途 | 状态 |
|--------|------|------|------|
| IVR | HV20 percentile rank (252D) | 波动率排位 | ✅ 已有 |
| HV_RATIO | HV20 / HV60 | 波动扩张/收缩 | ✅ 已有 |
| GARCH_VOL | GARCH(1,1) 预测波动 | 波动率预测 | 🆕 |
| TERM_STRUCTURE | HV20 - HV60 | 波动率期限结构 | 🆕 |
| REALIZED_SKEW | 日内高低差偏斜 | 方向性波动 | 🆕 |

#### E. 流动性因子 (新增 3 个)

| 因子名 | 公式 | 用途 |
|--------|------|------|
| AMIHUD | avg(|ret| / volume_dollar) | 非流动性 |
| SPREAD_PROXY | (High - Low) / Close | 买卖价差代理 |
| TURNOVER_STABILITY | std(turnover_20D, 60) | 换手稳定性 |

### 5.2 因子去冗余 (v1.1 新增)

40+ 因子中很多高度相关（如 MOM_1M/3M/6M/12M 之间 r > 0.7）。分两步筛选：

**Step 1: 相关性过滤**
- 计算因子间截面相关矩阵
- |corr| > 0.8 的因子对，保留 IC_IR 更高的那个
- 预计淘汰 ~10 个冗余因子

**Step 2: PCA 验证**
- 对保留的因子做 PCA
- 前 N 个主成分解释 90% 方差 → 实际有效维度 ~15-20
- 用 PCA 降维后的因子向量做后续分析（ChromaDB 也只存降维后的）

**最终有效因子预估**: 40+ 候选 → ~25 保留 → ~15 维 PCA 向量

### 5.3 因子计算管道

```python
class FactorPipeline:
    """每日/批量因子计算管道"""
    
    def run_daily(self):
        """增量更新: 只计算最新一天的因子值"""
        # 1. 下载今日行情 (yfinance)
        # 2. 追加到 prices.parquet
        # 3. 计算全部技术/风险/波动因子 (向量化, ~30秒)
        # 4. 写入 factors/*.parquet
        # 5. 更新向量数据库
    
    def run_full(self):
        """全量重算: 4500只 x 5年 (~5分钟)"""
        # 1. 加载全量 prices.parquet
        # 2. 计算所有因子
        # 3. 覆盖写入 factors/*.parquet
    
    def run_fundamental(self):
        """季度更新: 基本面因子 (~45分钟, API限速)"""
        # 1. 从 SecuritiesDB/yfinance 批量获取
        # 2. 计算衍生因子 (Piotroski F-Score 等)
        # 3. 写入 SQLite + fundamental.parquet
```

## 6. 向量检索层 (ChromaDB)

### 6.1 为什么用向量数据库？

传统做法：按单个因子排序 → 只能看一维
向量方法：把每只股票的 40+ 因子值组成一个向量 → 多维相似性搜索

**核心应用场景**:

| 场景 | 传统方法 | 向量方法 |
|------|---------|---------|
| "找和 NVDA 类似的股票" | 手动筛选行业+市值 | 一键查最近邻 |
| "哪些股票因子特征异常" | 逐个因子检查 | 聚类后找离群点 |
| "上次类似市场环境选了什么" | 凭记忆 | 历史截面向量搜索 |
| "哪些因子组合是冗余的" | IC相关矩阵 | 因子向量降维可视化 |

### 6.2 渐进式引入策略 (v1.1 优化)

> **自审结论**: 4500 只 x 40 维，numpy 的 cosine_similarity 只需毫秒。
> ChromaDB 在这个规模下是过度工程化。但如果后续要做：
> - 每日快照历史搜索 (4500 x 252天 x 5年 = 560万向量) → 需要向量索引
> - 多用户并发查询 → 需要持久化服务

**分步策略**:
1. **Phase 3a**: 先用 `scipy.spatial.distance` + numpy 实现相似搜索（0 依赖）
2. **Phase 3b**: 验证需求真实存在后，再引入 ChromaDB

**选择 ChromaDB 的理由** (Phase 3b 时评估):
- 纯 Python，无需外部服务 (`pip install chromadb`)
- 本地运行，数据不离开本机
- 支持 metadata 过滤 (按行业、市值分组筛选)
- 适合未来扩展到历史快照搜索 (百万级向量)

### 6.3 向量化设计

```python
# 每只股票 = 一个 40 维向量
stock_vector = [
    mom_1m_zscore, mom_3m_zscore, ...,    # 技术因子 (15维)
    ep_zscore, roe_zscore, ...,            # 基本面因子 (12维)
    beta_zscore, downvol_zscore, ...,      # 风险因子 (8维)
    ivr_zscore, hv_ratio_zscore, ...,      # 波动因子 (5维)
]

# 存入 ChromaDB
collection.add(
    ids=["AAPL_2026-05-05"],
    embeddings=[stock_vector],
    metadatas=[{
        "symbol": "AAPL",
        "date": "2026-05-05",
        "sector": "Technology",
        "market_cap": 3200000000000,
        "exchange": "NASDAQ",
    }]
)

# 查询: 找和 NVDA 最相似的 10 只股票
results = collection.query(
    query_embeddings=[nvda_vector],
    n_results=10,
    where={"sector": {"$ne": "Technology"}}  # 可跨行业找
)
```

### 6.4 高级应用

1. **时间序列向量**: 存储每日的市场截面向量 → 搜索"历史上哪天的市场和今天最像？"
2. **因子组合搜索**: "找一个因子组合，使得和我定义的 Alpha 信号最相关"
3. **风格漂移检测**: 监控持仓的因子向量随时间的变化，检测风格漂移

## 7. 实施计划

### Phase 1: 数据管道 (本次实施)

```
Week 1:
├── 1.1 获取全美股 ticker 列表 (GitHub JSON)
├── 1.2 批量下载行情数据 (yfinance, 4500只)
├── 1.3 Parquet 存储层 (价格数据)
└── 1.4 SQLite 元数据库 (universe 表)

Week 2:
├── 1.5 基本面数据采集 (yfinance + SecuritiesDB)
├── 1.6 增量更新机制 (每日自动)
└── 1.7 数据质量检查 (缺失值、异常值、退市处理)
```

### Phase 2: 因子引擎 (下一阶段)

```
├── 2.1 扩展技术因子 (15个)
├── 2.2 扩展基本面因子 (12个)
├── 2.3 新增风险因子 (8个)
├── 2.4 新增流动性因子 (3个)
├── 2.5 因子计算管道 (日更+季更)
└── 2.6 因子存储到 Parquet
```

### Phase 3: 向量检索 (后续)

```
├── 3.1 ChromaDB 集成
├── 3.2 股票相似性搜索
├── 3.3 历史截面搜索
├── 3.4 因子聚类和降维
└── 3.5 异常检测
```

### Phase 4: 策略整合 (最终)

```
├── 4.1 多因子选股引擎
├── 4.2 择时信号系统
├── 4.3 风控过滤器
└── 4.4 与现有策略 (Credit Spread / 动量轮动) 打通
```

## 8. 数据版本管理 (v1.1 新增)

因子值每天变化，因子计算公式也可能迭代。需要可追溯性：

```
data_store/factors/
├── technical.parquet                  # 当前版本
├── history/
│   ├── technical_v1_2026-05-05.parquet  # 公式变更时的快照
│   └── technical_v1_2026-06-01.parquet
└── meta.json                          # 因子版本元信息
```

`meta.json` 记录：
```json
{
  "version": "v1",
  "factors": ["MOM_1M", "MOM_3M", ...],
  "formulas_hash": "abc123",
  "last_computed": "2026-05-05",
  "row_count": 5700000
}
```

**规则**: 
- 日常增量更新覆盖 `technical.parquet`
- 因子公式变更时，旧版本保存到 `history/` 再重算
- 回测可指定因子版本

## 9. 技术依赖

```
# 新增依赖
chromadb        # 向量数据库
pyarrow         # Parquet 读写
requests        # API 调用 (已有)

# 已有依赖
pandas, numpy, scipy, yfinance, sqlite3
```

## 9. 性能预估

| 操作 | 数据量 | 预计耗时 |
|------|--------|---------|
| 全量下载行情 (yfinance) | 4500 只 x 5年 | ~25 分钟 |
| 增量更新行情 (每日) | 4500 只 x 1天 | ~3 分钟 |
| 全量因子计算 | 570万行 x 40因子 | ~5 分钟 |
| 增量因子计算 (每日) | 4500行 x 40因子 | ~10 秒 |
| 基本面数据采集 | 4500 只 | ~45 分钟 (API限速) |
| ChromaDB 全量索引 | 4500 x 40维 | ~3 秒 |
| ChromaDB 相似搜索 | 1 query | ~5 毫秒 |

## 11. 自审发现的问题和优化汇总 (v1.1)

| # | 问题 | 风险 | 优化措施 |
|---|------|------|---------|
| 1 | yfinance 单点依赖 | 被限速/封禁导致数据断流 | 分批+重试+备用源切换 |
| 2 | 40+ 因子冗余 | 多重共线性降低模型效果 | 相关性过滤+PCA去冗余 |
| 3 | ChromaDB 过度工程 | 增加复杂度无明显收益 | 先用 numpy，验证后再引入 |
| 4 | 存活偏差 | 回测结果过于乐观 | 只保留活跃股，回测报告标注偏差 |
| 8 | 拆股导致价格跳变 | 动量/波动因子失真 | 强制复权+入库时自动检测异常跳变 |
| 5 | Parquet 不分区 | 增量更新慢 | 按年分区 |
| 6 | 基本面串行下载 | 4500只需2小时+ | 10线程并行+分批 |
| 7 | 缺少版本管理 | 无法追溯因子公式变更 | meta.json+历史快照 |

## 12. 磁盘空间预估

| 文件 | 大小 |
|------|------|
| prices.parquet | ~80 MB |
| factors/*.parquet | ~120 MB |
| universe.db (SQLite) | ~5 MB |
| ChromaDB 索引 | ~20 MB |
| **总计** | **~225 MB** |
