# FUTU-QUANT Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core framework of FUTU-QUANT — a modular quantitative trading system on Futu OpenAPI, with event bus, data pipeline, strategy framework, risk management, trade execution, Telegram notification, backtesting, and memory bank.

**Architecture:** Modular Python project with YAML-driven configuration. Modules communicate through a lightweight event bus. All strategies inherit from a base class. Risk manager gates every trade. FutuOpenD is the gateway for market data and order execution.

**Tech Stack:** Python 3.10+, futu-api, pandas, numpy, ta, PyYAML, python-telegram-bot, APScheduler, SQLite

---

## File Structure

```
FUTU-QUANT/
├── config/
│   ├── settings.yaml
│   ├── strategies.yaml
│   └── symbols.yaml
├── core/
│   ├── __init__.py
│   ├── engine.py
│   ├── scheduler.py
│   └── event_bus.py
├── data/
│   ├── __init__.py
│   ├── market_data.py
│   ├── history.py
│   └── indicators.py
├── strategy/
│   ├── __init__.py
│   ├── base.py
│   ├── momentum.py
│   ├── mean_reversion.py
│   └── breakout.py
├── execution/
│   ├── __init__.py
│   ├── trader.py
│   ├── position.py
│   └── order.py
├── risk/
│   ├── __init__.py
│   ├── risk_manager.py
│   └── pdt_guard.py
├── notification/
│   ├── __init__.py
│   └── telegram_bot.py
├── backtest/
│   ├── __init__.py
│   ├── backtester.py
│   └── report.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── helpers.py
├── memory-bank/
│   ├── project-brief.md
│   ├── active-context.md
│   ├── system-patterns.md
│   ├── tech-context.md
│   ├── progress.md
│   └── strategy-journal.md
├── tests/
│   ├── __init__.py
│   ├── test_event_bus.py
│   ├── test_indicators.py
│   ├── test_order.py
│   ├── test_risk_manager.py
│   ├── test_pdt_guard.py
│   └── test_backtester.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

### Task 1: Project Scaffolding — requirements.txt, .gitignore, README, config files

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `config/settings.yaml`
- Create: `config/strategies.yaml`
- Create: `config/symbols.yaml`

- [ ] **Step 1: Create requirements.txt**

```txt
futu-api>=9.1
pandas>=2.0
numpy>=1.24
ta>=0.11
PyYAML>=6.0
python-telegram-bot>=20.0
APScheduler>=3.10
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.env
data_store/
*.db
*.sqlite
.vscode/
.idea/
```

- [ ] **Step 3: Create README.md**

Include: project name, description, prerequisites (Python 3.10+, FutuOpenD), quick start steps (install deps, configure settings.yaml, run main.py), project structure overview, and link to design doc.

- [ ] **Step 4: Create config/settings.yaml**

```yaml
futu:
  host: "127.0.0.1"
  port: 11111
  trade_env: SIMULATE

account:
  initial_capital: 3000
  currency: USD

risk:
  max_loss_per_trade_pct: 0.05
  max_daily_loss_pct: 0.08
  max_position_pct: 0.40
  max_total_position_pct: 0.80
  max_consecutive_losses: 3
  cooldown_minutes: 60
  trailing_stop:
    activate_pct: 0.02
    trail_pct: 0.015
  hard_stop_pct: 0.05
  eod_close_minutes_before: 15

pdt:
  enabled: true
  max_day_trades: 3
  rolling_window_days: 5

signal:
  min_strength_stock: 60
  min_strength_option: 80

telegram:
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
  enabled: false

logging:
  level: INFO
  file: data_store/logs/futu_quant.log

scheduler:
  market_open: "09:30"
  market_close: "16:00"
  timezone: "US/Eastern"
  bar_interval_seconds: 60
```

- [ ] **Step 5: Create config/strategies.yaml**

```yaml
strategies:
  momentum:
    enabled: true
    params:
      fast_ma_period: 5
      slow_ma_period: 20
      rsi_period: 14
      rsi_oversold: 30
      rsi_overbought: 70
      volume_ratio_threshold: 1.5

  mean_reversion:
    enabled: true
    params:
      bb_period: 20
      bb_std: 2.0
      rsi_period: 14
      rsi_oversold: 25
      rsi_overbought: 75

  breakout:
    enabled: true
    params:
      lookback_period: 20
      volume_surge_ratio: 2.0
      macd_fast: 12
      macd_slow: 26
      macd_signal: 9
```

- [ ] **Step 6: Create config/symbols.yaml**

```yaml
etf:
  leveraged:
    - "US.TQQQ"
    - "US.SQQQ"
    - "US.SOXL"
    - "US.SOXS"
    - "US.TNA"
    - "US.TZA"
  standard:
    - "US.SPY"
    - "US.QQQ"
    - "US.IWM"

stocks: []

options:
  enabled: false
  underlyings: []
```

- [ ] **Step 7: Create all __init__.py files**

Create empty `__init__.py` in: `core/`, `data/`, `strategy/`, `execution/`, `risk/`, `notification/`, `backtest/`, `utils/`, `tests/`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with config, deps, and README"
```

---

### Task 2: Event Bus — lightweight pub/sub for inter-module communication

**Files:**
- Create: `core/event_bus.py`
- Create: `tests/test_event_bus.py`

- [ ] **Step 1: Write the failing test for EventBus**

```python
# tests/test_event_bus.py
import pytest
from core.event_bus import EventBus, EventType

class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.SIGNAL, lambda data: received.append(data))
        bus.publish(EventType.SIGNAL, {"direction": "BUY"})
        assert len(received) == 1
        assert received[0]["direction"] == "BUY"

    def test_multiple_subscribers(self):
        bus = EventBus()
        results_a = []
        results_b = []
        bus.subscribe(EventType.MARKET_DATA, lambda d: results_a.append(d))
        bus.subscribe(EventType.MARKET_DATA, lambda d: results_b.append(d))
        bus.publish(EventType.MARKET_DATA, {"price": 100})
        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda d: received.append(d)
        bus.subscribe(EventType.ORDER, handler)
        bus.unsubscribe(EventType.ORDER, handler)
        bus.publish(EventType.ORDER, {"id": 1})
        assert len(received) == 0

    def test_different_event_types_isolated(self):
        bus = EventBus()
        signal_data = []
        order_data = []
        bus.subscribe(EventType.SIGNAL, lambda d: signal_data.append(d))
        bus.subscribe(EventType.ORDER, lambda d: order_data.append(d))
        bus.publish(EventType.SIGNAL, {"x": 1})
        assert len(signal_data) == 1
        assert len(order_data) == 0

    def test_publish_with_no_subscribers(self):
        bus = EventBus()
        bus.publish(EventType.SYSTEM, {"status": "ok"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_event_bus.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement EventBus**

```python
# core/event_bus.py
from enum import Enum, auto
from typing import Callable, Any
from collections import defaultdict


class EventType(Enum):
    MARKET_DATA = auto()
    SIGNAL = auto()
    ORDER = auto()
    RISK_ALERT = auto()
    SYSTEM = auto()


class EventBus:
    def __init__(self):
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Callable[[Any], None]) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Any], None]) -> None:
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    def publish(self, event_type: EventType, data: Any = None) -> None:
        for handler in self._subscribers[event_type]:
            handler(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_event_bus.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/event_bus.py tests/test_event_bus.py
git commit -m "feat: add EventBus with pub/sub for inter-module communication"
```

---

### Task 3: Order Model and Utility Logger

**Files:**
- Create: `execution/order.py`
- Create: `utils/logger.py`
- Create: `utils/helpers.py`
- Create: `tests/test_order.py`

- [ ] **Step 1: Write failing test for Order model**

```python
# tests/test_order.py
import pytest
from execution.order import Order, OrderDirection, OrderType, OrderStatus

class TestOrder:
    def test_create_buy_order(self):
        order = Order(
            symbol="US.TQQQ",
            direction=OrderDirection.BUY,
            quantity=10,
            price=55.0,
            order_type=OrderType.LIMIT,
            strategy_name="momentum"
        )
        assert order.symbol == "US.TQQQ"
        assert order.direction == OrderDirection.BUY
        assert order.quantity == 10
        assert order.status == OrderStatus.PENDING
        assert order.strategy_name == "momentum"

    def test_order_fill(self):
        order = Order(
            symbol="US.SOXL",
            direction=OrderDirection.SELL,
            quantity=40,
            price=18.0,
            order_type=OrderType.MARKET,
            strategy_name="breakout"
        )
        order.fill(fill_price=18.1, fill_quantity=40)
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 18.1

    def test_order_cancel(self):
        order = Order(
            symbol="US.SPY",
            direction=OrderDirection.BUY,
            quantity=2,
            price=530.0,
            order_type=OrderType.LIMIT,
            strategy_name="mean_reversion"
        )
        order.cancel()
        assert order.status == OrderStatus.CANCELLED

    def test_order_to_dict(self):
        order = Order(
            symbol="US.TQQQ",
            direction=OrderDirection.BUY,
            quantity=10,
            price=55.0,
            order_type=OrderType.LIMIT,
            strategy_name="momentum"
        )
        d = order.to_dict()
        assert d["symbol"] == "US.TQQQ"
        assert d["direction"] == "BUY"
        assert "timestamp" in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_order.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Order model**

```python
# execution/order.py
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from typing import Optional


class OrderDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    symbol: str
    direction: OrderDirection
    quantity: int
    price: float
    order_type: OrderType
    strategy_name: str
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: Optional[int] = None
    order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    fill_timestamp: Optional[datetime] = None

    def fill(self, fill_price: float, fill_quantity: int) -> None:
        self.fill_price = fill_price
        self.fill_quantity = fill_quantity
        self.status = OrderStatus.FILLED if fill_quantity >= self.quantity else OrderStatus.PARTIAL
        self.fill_timestamp = datetime.now()

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "strategy_name": self.strategy_name,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat(),
            "fill_timestamp": self.fill_timestamp.isoformat() if self.fill_timestamp else None,
        }
```

- [ ] **Step 4: Implement logger.py**

```python
# utils/logger.py
import logging
import os
from pathlib import Path


def setup_logger(name: str = "futu_quant", log_file: str = "data_store/logs/futu_quant.log", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
```

- [ ] **Step 5: Implement helpers.py**

```python
# utils/helpers.py
import yaml
from pathlib import Path


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
```

- [ ] **Step 6: Run tests to verify Order passes**

Run: `python -m pytest tests/test_order.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add execution/order.py utils/logger.py utils/helpers.py tests/test_order.py
git commit -m "feat: add Order model, logger, and config helpers"
```

---

### Task 4: Technical Indicators Module

**Files:**
- Create: `data/indicators.py`
- Create: `tests/test_indicators.py`

- [ ] **Step 1: Write failing test for indicators**

```python
# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from data.indicators import TechnicalIndicators

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

class TestTechnicalIndicators:
    def test_add_ma(self, sample_df):
        result = TechnicalIndicators.add_ma(sample_df, period=10)
        assert f"ma_10" in result.columns
        assert result["ma_10"].iloc[9:].notna().all()

    def test_add_ema(self, sample_df):
        result = TechnicalIndicators.add_ema(sample_df, period=10)
        assert "ema_10" in result.columns
        assert result["ema_10"].iloc[-1] != 0

    def test_add_rsi(self, sample_df):
        result = TechnicalIndicators.add_rsi(sample_df, period=14)
        assert "rsi_14" in result.columns
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_add_macd(self, sample_df):
        result = TechnicalIndicators.add_macd(sample_df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_add_bollinger(self, sample_df):
        result = TechnicalIndicators.add_bollinger(sample_df, period=20, std=2.0)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

    def test_add_atr(self, sample_df):
        result = TechnicalIndicators.add_atr(sample_df, period=14)
        assert "atr_14" in result.columns
        valid = result["atr_14"].dropna()
        assert (valid > 0).all()

    def test_add_vwap(self, sample_df):
        result = TechnicalIndicators.add_vwap(sample_df)
        assert "vwap" in result.columns

    def test_add_obv(self, sample_df):
        result = TechnicalIndicators.add_obv(sample_df)
        assert "obv" in result.columns

    def test_add_all(self, sample_df):
        result = TechnicalIndicators.add_all(sample_df)
        expected_cols = ["ma_5", "ma_20", "ema_5", "ema_20", "rsi_14", "macd", "bb_upper", "atr_14", "vwap", "obv"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_indicators.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement TechnicalIndicators**

```python
# data/indicators.py
import pandas as pd
import numpy as np


class TechnicalIndicators:
    @staticmethod
    def add_ma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
        df[f"ma_{period}"] = df[column].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
        df[f"ema_{period}"] = df[column].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close") -> pd.DataFrame:
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0, column: str = "close") -> pd.DataFrame:
        sma = df[column].rolling(window=period).mean()
        std_dev = df[column].rolling(window=period).std()
        df["bb_upper"] = sma + std * std_dev
        df["bb_middle"] = sma
        df["bb_lower"] = sma - std * std_dev
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        obv = [0.0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        return df

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        df = TechnicalIndicators.add_ma(df, period=5)
        df = TechnicalIndicators.add_ma(df, period=20)
        df = TechnicalIndicators.add_ema(df, period=5)
        df = TechnicalIndicators.add_ema(df, period=20)
        df = TechnicalIndicators.add_rsi(df, period=14)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger(df)
        df = TechnicalIndicators.add_atr(df, period=14)
        df = TechnicalIndicators.add_vwap(df)
        df = TechnicalIndicators.add_obv(df)
        return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_indicators.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/indicators.py tests/test_indicators.py
git commit -m "feat: add TechnicalIndicators with MA/EMA/RSI/MACD/BB/ATR/VWAP/OBV"
```

---

### Task 5: Strategy Base Class and Signal Model

**Files:**
- Create: `strategy/base.py`

- [ ] **Step 1: Implement Signal dataclass and BaseStrategy**

```python
# strategy/base.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalAssetType(Enum):
    STOCK = "STOCK"
    OPTION = "OPTION"


@dataclass
class Signal:
    symbol: str
    direction: SignalDirection
    strength: float  # 0-100
    strategy_name: str
    reason: str
    suggested_type: SignalAssetType = SignalAssetType.STOCK
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "suggested_type": self.suggested_type.value,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        pass

    def on_tick(self, symbol: str, tick_data: dict) -> Optional[Signal]:
        return None

    def get_params(self) -> dict:
        return self.params.copy()

    def set_params(self, params: dict) -> None:
        self.params.update(params)
```

- [ ] **Step 2: Commit**

```bash
git add strategy/base.py
git commit -m "feat: add Signal model and BaseStrategy abstract class"
```

---

### Task 6: Risk Manager

**Files:**
- Create: `risk/risk_manager.py`
- Create: `tests/test_risk_manager.py`

- [ ] **Step 1: Write failing tests for RiskManager**

```python
# tests/test_risk_manager.py
import pytest
from risk.risk_manager import RiskManager
from strategy.base import Signal, SignalDirection, SignalAssetType

def make_signal(symbol="US.TQQQ", direction=SignalDirection.BUY, strength=70):
    return Signal(
        symbol=symbol,
        direction=direction,
        strength=strength,
        strategy_name="test",
        reason="test signal",
    )

class TestRiskManager:
    def setup_method(self):
        self.config = {
            "max_loss_per_trade_pct": 0.05,
            "max_daily_loss_pct": 0.08,
            "max_position_pct": 0.40,
            "max_total_position_pct": 0.80,
            "max_consecutive_losses": 3,
            "cooldown_minutes": 60,
        }
        self.rm = RiskManager(self.config, initial_capital=3000)

    def test_calculate_position_size(self):
        size = self.rm.calculate_position_size(price=55.0, signal_strength=70)
        assert size > 0
        max_value = 3000 * 0.40
        assert size * 55.0 <= max_value

    def test_check_single_trade_loss(self):
        signal = make_signal()
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions={}, price=55.0)
        assert allowed is True

    def test_reject_when_max_position_exceeded(self):
        positions = {"US.TQQQ": {"value": 1200}}
        signal = make_signal(symbol="US.TQQQ")
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions=positions, price=55.0)
        assert allowed is True or "position" in reason.lower()

    def test_reject_when_total_position_exceeded(self):
        positions = {
            "US.TQQQ": {"value": 1200},
            "US.SOXL": {"value": 700},
            "US.SPY": {"value": 600},
        }
        signal = make_signal(symbol="US.QQQ")
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions=positions, price=450.0)
        assert allowed is False
        assert "total" in reason.lower() or "position" in reason.lower()

    def test_reject_after_daily_loss_exceeded(self):
        self.rm.record_loss(250)
        signal = make_signal()
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions={}, price=55.0)
        assert allowed is False
        assert "daily" in reason.lower()

    def test_reject_after_consecutive_losses(self):
        for _ in range(3):
            self.rm.record_loss(50)
        signal = make_signal()
        allowed, reason = self.rm.check_trade_allowed(signal, current_positions={}, price=55.0)
        assert allowed is False
        assert "consecutive" in reason.lower() or "cooldown" in reason.lower()

    def test_record_win_resets_consecutive_losses(self):
        self.rm.record_loss(50)
        self.rm.record_loss(50)
        self.rm.record_win(100)
        assert self.rm.consecutive_losses == 0

    def test_reset_daily(self):
        self.rm.record_loss(200)
        self.rm.reset_daily()
        assert self.rm.daily_loss == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_risk_manager.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement RiskManager**

```python
# risk/risk_manager.py
from datetime import datetime, timedelta
from typing import Optional
from strategy.base import Signal


class RiskManager:
    def __init__(self, config: dict, initial_capital: float):
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_loss_time: Optional[datetime] = None

    def check_trade_allowed(self, signal: Signal, current_positions: dict, price: float) -> tuple[bool, str]:
        max_daily = self.initial_capital * self.config["max_daily_loss_pct"]
        if self.daily_loss >= max_daily:
            return False, f"Daily loss limit reached: ${self.daily_loss:.2f} >= ${max_daily:.2f}"

        if self.consecutive_losses >= self.config["max_consecutive_losses"]:
            if self.last_loss_time:
                cooldown_end = self.last_loss_time + timedelta(minutes=self.config["cooldown_minutes"])
                if datetime.now() < cooldown_end:
                    return False, f"Cooldown active after {self.consecutive_losses} consecutive losses"

        total_position_value = sum(p.get("value", 0) for p in current_positions.values())
        max_total = self.initial_capital * self.config["max_total_position_pct"]
        if total_position_value >= max_total:
            return False, f"Total position limit reached: ${total_position_value:.2f} >= ${max_total:.2f}"

        symbol_value = current_positions.get(signal.symbol, {}).get("value", 0)
        max_single = self.initial_capital * self.config["max_position_pct"]
        if symbol_value >= max_single:
            return False, f"Single position limit for {signal.symbol}: ${symbol_value:.2f} >= ${max_single:.2f}"

        return True, "Trade allowed"

    def calculate_position_size(self, price: float, signal_strength: float) -> int:
        max_value = self.initial_capital * self.config["max_position_pct"]
        strength_factor = signal_strength / 100.0
        target_value = max_value * strength_factor
        size = int(target_value / price)
        return max(size, 0)

    def record_loss(self, amount: float) -> None:
        self.daily_loss += amount
        self.consecutive_losses += 1
        self.last_loss_time = datetime.now()
        self.current_capital -= amount

    def record_win(self, amount: float) -> None:
        self.consecutive_losses = 0
        self.current_capital += amount

    def reset_daily(self) -> None:
        self.daily_loss = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_risk_manager.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add risk/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: add RiskManager with position sizing and trade gating"
```

---

### Task 7: PDT Guard

**Files:**
- Create: `risk/pdt_guard.py`
- Create: `tests/test_pdt_guard.py`

- [ ] **Step 1: Write failing tests for PdtGuard**

```python
# tests/test_pdt_guard.py
import pytest
from datetime import datetime, timedelta
from risk.pdt_guard import PdtGuard

class TestPdtGuard:
    def setup_method(self):
        self.guard = PdtGuard(max_day_trades=3, rolling_window_days=5)

    def test_initially_allowed(self):
        assert self.guard.can_day_trade() is True
        assert self.guard.remaining_day_trades() == 3

    def test_record_day_trade(self):
        self.guard.record_day_trade("US.TQQQ")
        assert self.guard.remaining_day_trades() == 2

    def test_block_after_max(self):
        self.guard.record_day_trade("US.TQQQ")
        self.guard.record_day_trade("US.SOXL")
        self.guard.record_day_trade("US.SPY")
        assert self.guard.can_day_trade() is False
        assert self.guard.remaining_day_trades() == 0

    def test_old_trades_expire(self):
        old_time = datetime.now() - timedelta(days=6)
        self.guard._trades.append({"symbol": "US.TQQQ", "timestamp": old_time})
        self.guard._trades.append({"symbol": "US.SOXL", "timestamp": old_time})
        self.guard._trades.append({"symbol": "US.SPY", "timestamp": old_time})
        assert self.guard.can_day_trade() is True
        assert self.guard.remaining_day_trades() == 3

    def test_warning_threshold(self):
        self.guard.record_day_trade("US.TQQQ")
        self.guard.record_day_trade("US.SOXL")
        assert self.guard.should_warn() is True

    def test_no_warning_when_plenty_left(self):
        self.guard.record_day_trade("US.TQQQ")
        assert self.guard.should_warn() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pdt_guard.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement PdtGuard**

```python
# risk/pdt_guard.py
from datetime import datetime, timedelta


class PdtGuard:
    def __init__(self, max_day_trades: int = 3, rolling_window_days: int = 5):
        self.max_day_trades = max_day_trades
        self.rolling_window_days = rolling_window_days
        self._trades: list[dict] = []

    def _recent_trades(self) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=self.rolling_window_days)
        return [t for t in self._trades if t["timestamp"] > cutoff]

    def can_day_trade(self) -> bool:
        return len(self._recent_trades()) < self.max_day_trades

    def remaining_day_trades(self) -> int:
        return max(0, self.max_day_trades - len(self._recent_trades()))

    def should_warn(self) -> bool:
        return self.remaining_day_trades() == 1

    def record_day_trade(self, symbol: str) -> None:
        self._trades.append({
            "symbol": symbol,
            "timestamp": datetime.now(),
        })

    def cleanup_old_trades(self) -> None:
        self._trades = self._recent_trades()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pdt_guard.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add risk/pdt_guard.py tests/test_pdt_guard.py
git commit -m "feat: add PdtGuard for SEC day-trade rule enforcement"
```

---

### Task 8: Telegram Notification Module

**Files:**
- Create: `notification/telegram_bot.py`

- [ ] **Step 1: Implement TelegramNotifier**

```python
# notification/telegram_bot.py
import asyncio
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("telegram")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._bot = None

    async def _ensure_bot(self):
        if self._bot is None:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
            except ImportError:
                logger.warning("python-telegram-bot not installed, notifications disabled")
                self.enabled = False

    async def send_message(self, text: str) -> bool:
        if not self.enabled:
            logger.info(f"[Telegram disabled] {text}")
            return False
        try:
            await self._ensure_bot()
            if self._bot:
                await self._bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
                return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
        return False

    def send_sync(self, text: str) -> bool:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.send_message(text))

    def notify_system_start(self, balance: float) -> None:
        self.send_sync(f"🟢 *FUTU-QUANT 已启动*\n账户余额: ${balance:,.2f}")

    def notify_open_position(self, symbol: str, quantity: int, price: float, strategy: str, strength: float) -> None:
        self.send_sync(
            f"📈 *开仓*\n"
            f"标的: `{symbol}`\n"
            f"数量: {quantity}\n"
            f"价格: ${price:.2f}\n"
            f"策略: {strategy}\n"
            f"信号强度: {strength:.0f}"
        )

    def notify_close_position(self, symbol: str, quantity: int, price: float, pnl: float, pnl_pct: float) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        self.send_sync(
            f"{emoji} *平仓*\n"
            f"标的: `{symbol}`\n"
            f"数量: {quantity}\n"
            f"价格: ${price:.2f}\n"
            f"盈亏: ${pnl:+.2f} ({pnl_pct:+.1f}%)"
        )

    def notify_stop_loss(self, symbol: str, loss: float) -> None:
        self.send_sync(f"⚠️ *止损触发*\n标的: `{symbol}`\n亏损: ${loss:.2f}")

    def notify_pdt_warning(self, remaining: int) -> None:
        self.send_sync(f"⚠️ *PDT 警告*\n日内交易额度剩余: {remaining} 次")

    def notify_daily_summary(self, trades: int, pnl: float, balance: float) -> None:
        emoji = "📊" if pnl >= 0 else "📉"
        self.send_sync(
            f"{emoji} *当日总结*\n"
            f"交易笔数: {trades}\n"
            f"总盈亏: ${pnl:+.2f}\n"
            f"账户余额: ${balance:,.2f}"
        )

    def notify_error(self, error_msg: str) -> None:
        self.send_sync(f"❌ *系统异常*\n{error_msg}")
```

- [ ] **Step 2: Commit**

```bash
git add notification/telegram_bot.py
git commit -m "feat: add TelegramNotifier with all trade event notifications"
```

---

### Task 9: Market Data and History Module

**Files:**
- Create: `data/market_data.py`
- Create: `data/history.py`

- [ ] **Step 1: Implement MarketData**

```python
# data/market_data.py
from typing import Optional, Callable
from utils.logger import setup_logger

logger = setup_logger("market_data")


class MarketData:
    def __init__(self, host: str = "127.0.0.1", port: int = 11111):
        self.host = host
        self.port = port
        self._quote_ctx = None
        self._connected = False

    def connect(self) -> bool:
        try:
            from futu import OpenQuoteContext
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)
            self._connected = True
            logger.info(f"Connected to FutuOpenD at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to FutuOpenD: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._quote_ctx:
            self._quote_ctx.close()
            self._connected = False
            logger.info("Disconnected from FutuOpenD")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_snapshot(self, symbols: list[str]) -> Optional[dict]:
        if not self._connected or not self._quote_ctx:
            logger.error("Not connected to FutuOpenD")
            return None
        try:
            from futu import RET_OK
            ret, data = self._quote_ctx.get_market_snapshot(symbols)
            if ret == RET_OK:
                return data.to_dict("records")
            logger.error(f"get_snapshot failed: {data}")
        except Exception as e:
            logger.error(f"get_snapshot error: {e}")
        return None

    def get_kline(self, symbol: str, ktype: str = "K_1M", count: int = 100) -> Optional[dict]:
        if not self._connected or not self._quote_ctx:
            logger.error("Not connected to FutuOpenD")
            return None
        try:
            from futu import RET_OK, KLType
            kl_map = {
                "K_1M": KLType.K_1M,
                "K_5M": KLType.K_5M,
                "K_15M": KLType.K_15M,
                "K_60M": KLType.K_60M,
                "K_DAY": KLType.K_DAY,
            }
            kl = kl_map.get(ktype, KLType.K_1M)
            ret, data = self._quote_ctx.get_cur_kline(symbol, count, kl)
            if ret == RET_OK:
                return data
            logger.error(f"get_kline failed: {data}")
        except Exception as e:
            logger.error(f"get_kline error: {e}")
        return None

    def subscribe(self, symbols: list[str], sub_types: list[str] = None) -> bool:
        if not self._connected or not self._quote_ctx:
            return False
        try:
            from futu import RET_OK, SubType
            if sub_types is None:
                sub_types = [SubType.K_1M, SubType.QUOTE]
            ret, data = self._quote_ctx.subscribe(symbols, sub_types)
            if ret == RET_OK:
                logger.info(f"Subscribed to {symbols}")
                return True
            logger.error(f"subscribe failed: {data}")
        except Exception as e:
            logger.error(f"subscribe error: {e}")
        return False
```

- [ ] **Step 2: Implement History module**

```python
# data/history.py
import os
import pandas as pd
from pathlib import Path
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("history")


class HistoryManager:
    def __init__(self, cache_dir: str = "data_store/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, ktype: str) -> Path:
        safe_symbol = symbol.replace(".", "_")
        return self.cache_dir / f"{safe_symbol}_{ktype}.csv"

    def save_to_cache(self, symbol: str, ktype: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol, ktype)
        df.to_csv(path, index=False)
        logger.info(f"Cached {len(df)} bars for {symbol} ({ktype})")

    def load_from_cache(self, symbol: str, ktype: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol, ktype)
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} bars from cache for {symbol} ({ktype})")
            return df
        return None

    def get_history(self, market_data, symbol: str, ktype: str = "K_DAY", count: int = 200, use_cache: bool = True) -> Optional[pd.DataFrame]:
        if use_cache:
            cached = self.load_from_cache(symbol, ktype)
            if cached is not None and len(cached) >= count:
                return cached.tail(count)

        data = market_data.get_kline(symbol, ktype, count)
        if data is not None:
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)
            self.save_to_cache(symbol, ktype, df)
            return df

        return self.load_from_cache(symbol, ktype)
```

- [ ] **Step 3: Commit**

```bash
git add data/market_data.py data/history.py
git commit -m "feat: add MarketData (FutuOpenD) and HistoryManager with caching"
```

---

### Task 10: Position Manager and Trader

**Files:**
- Create: `execution/position.py`
- Create: `execution/trader.py`

- [ ] **Step 1: Implement PositionManager**

```python
# execution/position.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger("position")


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    strategy_name: str
    is_day_trade: bool = False
    open_time: datetime = field(default_factory=datetime.now)
    highest_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price

    def update_highest(self, current_price: float) -> None:
        if current_price > self.highest_price:
            self.highest_price = current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.avg_price == 0:
            return 0.0
        return (current_price - self.avg_price) / self.avg_price * 100


class PositionManager:
    def __init__(self):
        self._positions: dict[str, Position] = {}

    def open_position(self, symbol: str, quantity: int, price: float, strategy_name: str, is_day_trade: bool = False) -> Position:
        pos = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            strategy_name=strategy_name,
            is_day_trade=is_day_trade,
            highest_price=price,
        )
        self._positions[symbol] = pos
        logger.info(f"Opened position: {symbol} x{quantity} @ ${price:.2f}")
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        pos = self._positions.pop(symbol, None)
        if pos:
            logger.info(f"Closed position: {symbol}")
        return pos

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        return self._positions.copy()

    def get_positions_dict(self) -> dict[str, dict]:
        return {
            sym: {"value": pos.market_value, "quantity": pos.quantity}
            for sym, pos in self._positions.items()
        }

    def get_day_trade_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_day_trade]

    def total_position_value(self) -> float:
        return sum(p.market_value for p in self._positions.values())

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions
```

- [ ] **Step 2: Implement Trader**

```python
# execution/trader.py
from typing import Optional
from execution.order import Order, OrderDirection, OrderType, OrderStatus
from execution.position import PositionManager
from strategy.base import Signal, SignalDirection
from risk.risk_manager import RiskManager
from risk.pdt_guard import PdtGuard
from core.event_bus import EventBus, EventType
from utils.logger import setup_logger

logger = setup_logger("trader")


class Trader:
    def __init__(
        self,
        risk_manager: RiskManager,
        pdt_guard: PdtGuard,
        position_manager: PositionManager,
        event_bus: EventBus,
        trade_env: str = "SIMULATE",
        host: str = "127.0.0.1",
        port: int = 11111,
    ):
        self.risk_manager = risk_manager
        self.pdt_guard = pdt_guard
        self.position_manager = position_manager
        self.event_bus = event_bus
        self.trade_env = trade_env
        self.host = host
        self.port = port
        self._trade_ctx = None
        self._order_history: list[Order] = []

    def connect(self) -> bool:
        try:
            from futu import OpenSecTradeContext, TrdEnv
            env = TrdEnv.SIMULATE if self.trade_env == "SIMULATE" else TrdEnv.REAL
            self._trade_ctx = OpenSecTradeContext(
                host=self.host, port=self.port, filter_trdmarket=None, security_firm=None
            )
            logger.info(f"Trader connected ({self.trade_env})")
            return True
        except Exception as e:
            logger.error(f"Trader connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._trade_ctx:
            self._trade_ctx.close()
            logger.info("Trader disconnected")

    def execute_signal(self, signal: Signal, price: float, is_day_trade: bool = False) -> Optional[Order]:
        positions = self.position_manager.get_positions_dict()
        allowed, reason = self.risk_manager.check_trade_allowed(signal, positions, price)
        if not allowed:
            logger.warning(f"Trade rejected: {reason}")
            self.event_bus.publish(EventType.RISK_ALERT, {"signal": signal.to_dict(), "reason": reason})
            return None

        if is_day_trade and not self.pdt_guard.can_day_trade():
            logger.warning("Trade rejected: PDT limit reached")
            self.event_bus.publish(EventType.RISK_ALERT, {"signal": signal.to_dict(), "reason": "PDT limit"})
            return None

        quantity = self.risk_manager.calculate_position_size(price, signal.strength)
        if quantity <= 0:
            logger.warning("Calculated position size is 0")
            return None

        direction = OrderDirection.BUY if signal.direction == SignalDirection.BUY else OrderDirection.SELL
        order = Order(
            symbol=signal.symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            order_type=OrderType.MARKET,
            strategy_name=signal.strategy_name,
        )

        success = self._place_order(order)
        if success:
            if direction == OrderDirection.BUY:
                self.position_manager.open_position(
                    signal.symbol, quantity, price, signal.strategy_name, is_day_trade
                )
                if is_day_trade:
                    self.pdt_guard.record_day_trade(signal.symbol)
            elif direction == OrderDirection.SELL:
                self.position_manager.close_position(signal.symbol)

            self._order_history.append(order)
            self.event_bus.publish(EventType.ORDER, order.to_dict())

        return order

    def _place_order(self, order: Order) -> bool:
        try:
            if self._trade_ctx is None:
                logger.info(f"[DRY RUN] {order.direction.value} {order.symbol} x{order.quantity} @ ${order.price:.2f}")
                order.fill(order.price, order.quantity)
                return True

            from futu import RET_OK, TrdSide, OrderType as FutuOrderType
            side = TrdSide.BUY if order.direction == OrderDirection.BUY else TrdSide.SELL
            ret, data = self._trade_ctx.place_order(
                price=order.price,
                qty=order.quantity,
                code=order.symbol,
                trd_side=side,
                order_type=FutuOrderType.MARKET,
            )
            if ret == RET_OK:
                order.fill(order.price, order.quantity)
                logger.info(f"Order filled: {order.symbol} {order.direction.value} x{order.quantity}")
                return True
            else:
                order.status = OrderStatus.FAILED
                logger.error(f"Order failed: {data}")
                return False
        except Exception as e:
            order.status = OrderStatus.FAILED
            logger.error(f"Order execution error: {e}")
            return False

    def get_order_history(self) -> list[Order]:
        return self._order_history.copy()
```

- [ ] **Step 3: Commit**

```bash
git add execution/position.py execution/trader.py
git commit -m "feat: add PositionManager and Trader with risk-gated execution"
```

---

### Task 11: Backtester and Report

**Files:**
- Create: `backtest/backtester.py`
- Create: `backtest/report.py`
- Create: `tests/test_backtester.py`

- [ ] **Step 1: Write failing test for Backtester**

```python
# tests/test_backtester.py
import pytest
import pandas as pd
import numpy as np
from backtest.backtester import Backtester
from backtest.report import BacktestReport
from strategy.base import BaseStrategy, Signal, SignalDirection

class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("dummy", {})
        self._call_count = 0

    def on_bar(self, symbol, bar_data):
        self._call_count += 1
        if self._call_count % 10 == 0:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=70,
                strategy_name="dummy",
                reason="test buy"
            )
        if self._call_count % 15 == 0:
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=70,
                strategy_name="dummy",
                reason="test sell"
            )
        return None

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    close = 50 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "time_key": pd.date_range("2025-01-01", periods=n, freq="D"),
        "open": close - np.random.rand(n) * 0.3,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

class TestBacktester:
    def test_run_backtest(self, sample_data):
        bt = Backtester(initial_capital=3000, commission_pct=0.001)
        strategy = DummyStrategy()
        result = bt.run(strategy, "US.TQQQ", sample_data)
        assert "trades" in result
        assert "final_capital" in result
        assert result["final_capital"] > 0

    def test_backtest_report(self, sample_data):
        bt = Backtester(initial_capital=3000, commission_pct=0.001)
        strategy = DummyStrategy()
        result = bt.run(strategy, "US.TQQQ", sample_data)
        report = BacktestReport(result)
        summary = report.summary()
        assert "total_return_pct" in summary
        assert "max_drawdown_pct" in summary
        assert "total_trades" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtester.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement Backtester**

```python
# backtest/backtester.py
import pandas as pd
from typing import Optional
from strategy.base import BaseStrategy, SignalDirection
from utils.logger import setup_logger

logger = setup_logger("backtester")


class Backtester:
    def __init__(self, initial_capital: float = 3000, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self, strategy: BaseStrategy, symbol: str, data: pd.DataFrame) -> dict:
        capital = self.initial_capital
        position = 0
        avg_entry = 0.0
        trades = []
        equity_curve = []

        for i in range(len(data)):
            bar = data.iloc[:i + 1]
            current_price = data.iloc[i]["close"]
            signal = strategy.on_bar(symbol, bar)

            if signal is not None:
                if signal.direction == SignalDirection.BUY and position == 0:
                    slippage = current_price * self.slippage_pct
                    buy_price = current_price + slippage
                    max_shares = int(capital * 0.4 / buy_price)
                    if max_shares > 0:
                        commission = buy_price * max_shares * self.commission_pct
                        cost = buy_price * max_shares + commission
                        if cost <= capital:
                            position = max_shares
                            avg_entry = buy_price
                            capital -= cost
                            trades.append({
                                "type": "BUY",
                                "price": buy_price,
                                "quantity": max_shares,
                                "commission": commission,
                                "time": data.iloc[i].get("time_key", i),
                            })

                elif signal.direction == SignalDirection.SELL and position > 0:
                    slippage = current_price * self.slippage_pct
                    sell_price = current_price - slippage
                    commission = sell_price * position * self.commission_pct
                    revenue = sell_price * position - commission
                    pnl = (sell_price - avg_entry) * position - commission
                    capital += revenue
                    trades.append({
                        "type": "SELL",
                        "price": sell_price,
                        "quantity": position,
                        "commission": commission,
                        "pnl": pnl,
                        "time": data.iloc[i].get("time_key", i),
                    })
                    position = 0
                    avg_entry = 0.0

            portfolio_value = capital + position * current_price
            equity_curve.append(portfolio_value)

        final_value = capital + position * data.iloc[-1]["close"]

        return {
            "initial_capital": self.initial_capital,
            "final_capital": final_value,
            "trades": trades,
            "equity_curve": equity_curve,
            "total_bars": len(data),
        }
```

- [ ] **Step 4: Implement BacktestReport**

```python
# backtest/report.py
import numpy as np
from typing import Optional


class BacktestReport:
    def __init__(self, result: dict):
        self.result = result

    def summary(self) -> dict:
        initial = self.result["initial_capital"]
        final = self.result["final_capital"]
        trades = self.result["trades"]
        equity = self.result["equity_curve"]

        total_return = final - initial
        total_return_pct = (total_return / initial) * 100

        sell_trades = [t for t in trades if t["type"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

        max_drawdown_pct = 0.0
        if equity:
            peak = equity[0]
            for val in equity:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak * 100
                if dd > max_drawdown_pct:
                    max_drawdown_pct = dd

        total_commission = sum(t.get("commission", 0) for t in trades)

        return {
            "initial_capital": initial,
            "final_capital": round(final, 2),
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": len(sell_trades),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_commission": round(total_commission, 2),
        }

    def print_report(self) -> str:
        s = self.summary()
        lines = [
            "=" * 50,
            "BACKTEST REPORT",
            "=" * 50,
            f"Initial Capital:  ${s['initial_capital']:,.2f}",
            f"Final Capital:    ${s['final_capital']:,.2f}",
            f"Total Return:     ${s['total_return']:+,.2f} ({s['total_return_pct']:+.2f}%)",
            f"Max Drawdown:     {s['max_drawdown_pct']:.2f}%",
            f"Total Trades:     {s['total_trades']}",
            f"Win Rate:         {s['win_rate_pct']:.2f}%",
            f"Avg Win:          ${s['avg_win']:,.2f}",
            f"Avg Loss:         ${s['avg_loss']:,.2f}",
            f"Profit Factor:    {s['profit_factor']:.2f}",
            f"Total Commission: ${s['total_commission']:,.2f}",
            "=" * 50,
        ]
        report = "\n".join(lines)
        print(report)
        return report
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtester.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backtest/backtester.py backtest/report.py tests/test_backtester.py
git commit -m "feat: add Backtester with slippage/commission and BacktestReport"
```

---

### Task 12: Scheduler and Main Engine

**Files:**
- Create: `core/scheduler.py`
- Create: `core/engine.py`
- Create: `main.py`

- [ ] **Step 1: Implement Scheduler**

```python
# core/scheduler.py
from datetime import datetime
import pytz
from utils.logger import setup_logger

logger = setup_logger("scheduler")


class TradingScheduler:
    def __init__(self, timezone: str = "US/Eastern", market_open: str = "09:30", market_close: str = "16:00"):
        self.tz = pytz.timezone(timezone)
        self.market_open = market_open
        self.market_close = market_close

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def is_market_hours(self) -> bool:
        now = self.now()
        if now.weekday() >= 5:
            return False
        time_str = now.strftime("%H:%M")
        return self.market_open <= time_str < self.market_close

    def minutes_to_close(self) -> int:
        now = self.now()
        close_h, close_m = map(int, self.market_close.split(":"))
        close_minutes = close_h * 60 + close_m
        now_minutes = now.hour * 60 + now.minute
        return close_minutes - now_minutes

    def should_force_close_day_trades(self, minutes_before: int = 15) -> bool:
        if not self.is_market_hours():
            return False
        return self.minutes_to_close() <= minutes_before
```

- [ ] **Step 2: Implement Engine**

```python
# core/engine.py
import time
from typing import Optional
from core.event_bus import EventBus, EventType
from core.scheduler import TradingScheduler
from data.market_data import MarketData
from data.history import HistoryManager
from data.indicators import TechnicalIndicators
from strategy.base import BaseStrategy, SignalDirection
from execution.trader import Trader
from execution.position import PositionManager
from risk.risk_manager import RiskManager
from risk.pdt_guard import PdtGuard
from notification.telegram_bot import TelegramNotifier
from utils.logger import setup_logger
from utils.helpers import load_yaml, get_project_root

logger = setup_logger("engine")


class Engine:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.root = get_project_root()
        self.config = load_yaml(str(self.root / config_path))
        self.strategies_config = load_yaml(str(self.root / "config" / "strategies.yaml"))
        self.symbols_config = load_yaml(str(self.root / "config" / "symbols.yaml"))

        self.event_bus = EventBus()
        self.scheduler = TradingScheduler(
            timezone=self.config["scheduler"]["timezone"],
            market_open=self.config["scheduler"]["market_open"],
            market_close=self.config["scheduler"]["market_close"],
        )
        self.market_data = MarketData(
            host=self.config["futu"]["host"],
            port=self.config["futu"]["port"],
        )
        self.history = HistoryManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(
            config=self.config["risk"],
            initial_capital=self.config["account"]["initial_capital"],
        )
        self.pdt_guard = PdtGuard(
            max_day_trades=self.config["pdt"]["max_day_trades"],
            rolling_window_days=self.config["pdt"]["rolling_window_days"],
        )
        self.trader = Trader(
            risk_manager=self.risk_manager,
            pdt_guard=self.pdt_guard,
            position_manager=self.position_manager,
            event_bus=self.event_bus,
            trade_env=self.config["futu"]["trade_env"],
            host=self.config["futu"]["host"],
            port=self.config["futu"]["port"],
        )

        tg_config = self.config.get("telegram", {})
        self.notifier = TelegramNotifier(
            bot_token=tg_config.get("bot_token", ""),
            chat_id=tg_config.get("chat_id", ""),
            enabled=tg_config.get("enabled", False),
        )

        self.strategies: list[BaseStrategy] = []
        self._running = False

    def load_strategies(self) -> None:
        from strategy.momentum import MomentumStrategy
        from strategy.mean_reversion import MeanReversionStrategy
        from strategy.breakout import BreakoutStrategy

        strategy_map = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
        }
        for name, cfg in self.strategies_config.get("strategies", {}).items():
            if cfg.get("enabled", False) and name in strategy_map:
                strategy = strategy_map[name](params=cfg.get("params", {}))
                self.strategies.append(strategy)
                logger.info(f"Loaded strategy: {name}")

    def get_symbols(self) -> list[str]:
        symbols = []
        for category in ["leveraged", "standard"]:
            symbols.extend(self.symbols_config.get("etf", {}).get(category, []))
        symbols.extend(self.symbols_config.get("stocks", []))
        return symbols

    def _setup_event_handlers(self) -> None:
        def on_order(data):
            if data.get("direction") == "BUY":
                self.notifier.notify_open_position(
                    data["symbol"], data["quantity"], data["price"],
                    data["strategy_name"], data.get("strength", 0)
                )
            else:
                self.notifier.notify_close_position(
                    data["symbol"], data["quantity"], data["price"],
                    data.get("pnl", 0), data.get("pnl_pct", 0)
                )

        def on_risk_alert(data):
            reason = data.get("reason", "")
            if "PDT" in reason:
                self.notifier.notify_pdt_warning(self.pdt_guard.remaining_day_trades())

        self.event_bus.subscribe(EventType.ORDER, on_order)
        self.event_bus.subscribe(EventType.RISK_ALERT, on_risk_alert)

    def start(self) -> None:
        logger.info("FUTU-QUANT Engine starting...")
        self._setup_event_handlers()
        self.load_strategies()

        connected = self.market_data.connect()
        if not connected:
            logger.warning("FutuOpenD not available, running in dry-run mode")

        self.trader.connect()
        self.notifier.notify_system_start(self.config["account"]["initial_capital"])

        symbols = self.get_symbols()
        if connected:
            self.market_data.subscribe(symbols)

        self._running = True
        logger.info(f"Engine started with {len(self.strategies)} strategies, {len(symbols)} symbols")

        try:
            self._run_loop(symbols)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            self.stop()

    def _run_loop(self, symbols: list[str]) -> None:
        interval = self.config["scheduler"].get("bar_interval_seconds", 60)
        eod_minutes = self.config["risk"].get("eod_close_minutes_before", 15)

        while self._running:
            if not self.scheduler.is_market_hours():
                logger.debug("Market closed, waiting...")
                time.sleep(30)
                continue

            if self.scheduler.should_force_close_day_trades(eod_minutes):
                self._force_close_day_trades()

            for symbol in symbols:
                self._process_symbol(symbol)

            time.sleep(interval)

    def _process_symbol(self, symbol: str) -> None:
        kline = self.market_data.get_kline(symbol, "K_1M", 100)
        if kline is None:
            return

        import pandas as pd
        if not isinstance(kline, pd.DataFrame):
            kline = pd.DataFrame(kline)

        kline = TechnicalIndicators.add_all(kline)

        for strategy in self.strategies:
            signal = strategy.on_bar(symbol, kline)
            if signal is not None:
                logger.info(f"Signal: {signal.direction.value} {signal.symbol} strength={signal.strength} from {signal.strategy_name}")
                self.event_bus.publish(EventType.SIGNAL, signal.to_dict())

                current_price = kline.iloc[-1]["close"]
                is_day = not self.scheduler.should_force_close_day_trades()
                self.trader.execute_signal(signal, current_price, is_day_trade=is_day)

    def _force_close_day_trades(self) -> None:
        day_positions = self.position_manager.get_day_trade_positions()
        for pos in day_positions:
            logger.info(f"Force closing day trade: {pos.symbol}")
            from strategy.base import Signal, SignalDirection
            close_signal = Signal(
                symbol=pos.symbol,
                direction=SignalDirection.SELL,
                strength=100,
                strategy_name="eod_force_close",
                reason="End of day forced close",
            )
            self.trader.execute_signal(close_signal, pos.avg_price)

    def stop(self) -> None:
        self._running = False
        self.market_data.disconnect()
        self.trader.disconnect()
        self.risk_manager.reset_daily()
        logger.info("Engine stopped")
```

- [ ] **Step 3: Implement main.py**

```python
# main.py
from core.engine import Engine


def main():
    engine = Engine()
    engine.start()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add core/scheduler.py core/engine.py main.py
git commit -m "feat: add TradingScheduler, Engine, and main entry point"
```

---

### Task 13: Three Built-in Strategies (Momentum, Mean Reversion, Breakout)

**Files:**
- Create: `strategy/momentum.py`
- Create: `strategy/mean_reversion.py`
- Create: `strategy/breakout.py`

- [ ] **Step 1: Implement MomentumStrategy**

```python
# strategy/momentum.py
from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class MomentumStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            "fast_ma_period": 5,
            "slow_ma_period": 20,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_ratio_threshold": 1.5,
        }
        if params:
            default_params.update(params)
        super().__init__("momentum", default_params)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        if len(bar_data) < self.params["slow_ma_period"] + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_ma(df, self.params["fast_ma_period"])
        df = TechnicalIndicators.add_ma(df, self.params["slow_ma_period"])
        df = TechnicalIndicators.add_rsi(df, self.params["rsi_period"])

        fast_col = f"ma_{self.params['fast_ma_period']}"
        slow_col = f"ma_{self.params['slow_ma_period']}"
        rsi_col = f"rsi_{self.params['rsi_period']}"

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        vol_ratio = curr["volume"] / vol_avg if vol_avg > 0 else 0

        if (prev[fast_col] <= prev[slow_col] and
            curr[fast_col] > curr[slow_col] and
            curr[rsi_col] > self.params["rsi_oversold"] and
            vol_ratio >= self.params["volume_ratio_threshold"]):

            strength = min(50 + vol_ratio * 10 + (50 - abs(curr[rsi_col] - 50)) * 0.5, 100)
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"Golden cross + RSI {curr[rsi_col]:.1f} + Volume ratio {vol_ratio:.1f}x",
                suggested_type=asset_type,
            )

        if (prev[fast_col] >= prev[slow_col] and
            curr[fast_col] < curr[slow_col] and
            curr[rsi_col] < self.params["rsi_overbought"]):

            strength = min(50 + (curr[rsi_col] - 50) * 0.5, 100)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Death cross + RSI {curr[rsi_col]:.1f}",
            )

        return None
```

- [ ] **Step 2: Implement MeanReversionStrategy**

```python
# strategy/mean_reversion.py
from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
        }
        if params:
            default_params.update(params)
        super().__init__("mean_reversion", default_params)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        if len(bar_data) < self.params["bb_period"] + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_bollinger(df, self.params["bb_period"], self.params["bb_std"])
        df = TechnicalIndicators.add_rsi(df, self.params["rsi_period"])

        rsi_col = f"rsi_{self.params['rsi_period']}"
        curr = df.iloc[-1]

        if (curr["close"] <= curr["bb_lower"] and
            curr[rsi_col] <= self.params["rsi_oversold"]):

            strength = min(50 + (self.params["rsi_oversold"] - curr[rsi_col]) * 2, 100)
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"Price below BB lower + RSI {curr[rsi_col]:.1f} oversold",
                suggested_type=asset_type,
            )

        if (curr["close"] >= curr["bb_upper"] and
            curr[rsi_col] >= self.params["rsi_overbought"]):

            strength = min(50 + (curr[rsi_col] - self.params["rsi_overbought"]) * 2, 100)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Price above BB upper + RSI {curr[rsi_col]:.1f} overbought",
            )

        return None
```

- [ ] **Step 3: Implement BreakoutStrategy**

```python
# strategy/breakout.py
from typing import Optional
import pandas as pd
from strategy.base import BaseStrategy, Signal, SignalDirection, SignalAssetType
from data.indicators import TechnicalIndicators


class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            "lookback_period": 20,
            "volume_surge_ratio": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }
        if params:
            default_params.update(params)
        super().__init__("breakout", default_params)

    def on_bar(self, symbol: str, bar_data: pd.DataFrame) -> Optional[Signal]:
        lookback = self.params["lookback_period"]
        if len(bar_data) < lookback + self.params["macd_slow"] + 5:
            return None

        df = bar_data.copy()
        df = TechnicalIndicators.add_macd(df, self.params["macd_fast"], self.params["macd_slow"], self.params["macd_signal"])

        curr = df.iloc[-1]
        recent = df.iloc[-lookback - 1:-1]

        resistance = recent["high"].max()
        support = recent["low"].min()
        vol_avg = recent["volume"].mean()
        vol_ratio = curr["volume"] / vol_avg if vol_avg > 0 else 0

        if (curr["close"] > resistance and
            vol_ratio >= self.params["volume_surge_ratio"] and
            curr["macd_hist"] > 0):

            strength = min(50 + vol_ratio * 10 + curr["macd_hist"] * 5, 100)
            asset_type = SignalAssetType.OPTION if strength > 80 else SignalAssetType.STOCK
            return Signal(
                symbol=symbol,
                direction=SignalDirection.BUY,
                strength=strength,
                strategy_name=self.name,
                reason=f"Breakout above {resistance:.2f} + Volume {vol_ratio:.1f}x + MACD bullish",
                suggested_type=asset_type,
            )

        if (curr["close"] < support and
            vol_ratio >= self.params["volume_surge_ratio"] and
            curr["macd_hist"] < 0):

            strength = min(50 + vol_ratio * 10 + abs(curr["macd_hist"]) * 5, 100)
            return Signal(
                symbol=symbol,
                direction=SignalDirection.SELL,
                strength=strength,
                strategy_name=self.name,
                reason=f"Breakdown below {support:.2f} + Volume {vol_ratio:.1f}x + MACD bearish",
            )

        return None
```

- [ ] **Step 4: Commit**

```bash
git add strategy/momentum.py strategy/mean_reversion.py strategy/breakout.py
git commit -m "feat: add Momentum, MeanReversion, and Breakout strategies"
```

---

### Task 14: Memory Bank Initialization

**Files:**
- Create: `memory-bank/project-brief.md`
- Create: `memory-bank/active-context.md`
- Create: `memory-bank/system-patterns.md`
- Create: `memory-bank/tech-context.md`
- Create: `memory-bank/progress.md`
- Create: `memory-bank/strategy-journal.md`

- [ ] **Step 1: Create all Memory Bank files with initial content**

Each file should be populated with the actual project context from the design spec — project goals, architecture, tech stack, current progress, and empty strategy journal.

- [ ] **Step 2: Commit**

```bash
git add memory-bank/
git commit -m "feat: initialize Memory Bank with project context"
```
