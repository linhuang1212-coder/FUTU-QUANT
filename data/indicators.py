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

    @staticmethod
    def resample_kline(
        df: pd.DataFrame,
        rule: str,
        time_col: str = "time_key",
    ) -> pd.DataFrame:
        """Resample OHLCV data to a coarser timeframe.

        Args:
            df: DataFrame with OHLCV columns and a datetime-like *time_col*.
            rule: A pandas offset alias, e.g. '5min', '15min', '1h', '1W', '1M'.
            time_col: Name of the datetime column (will be used as the resample
                      index and restored afterwards).

        Returns:
            A new DataFrame resampled to *rule* with proper OHLCV aggregation.
            Rows where volume == 0 (no trading) are dropped.
        """
        src = df.copy()
        src[time_col] = pd.to_datetime(src[time_col])
        src = src.set_index(time_col).sort_index()

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        available = {c: agg[c] for c in agg if c in src.columns}
        resampled = src.resample(rule).agg(available).dropna(subset=["close"])

        if "volume" in resampled.columns:
            resampled = resampled[resampled["volume"] > 0]

        resampled = resampled.reset_index().rename(columns={"index": time_col})
        if time_col not in resampled.columns and resampled.index.name == time_col:
            resampled = resampled.reset_index()

        return resampled
