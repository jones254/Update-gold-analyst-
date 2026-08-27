"""
Data source abstraction for the Gold Scalper engine.

Two providers, single interface:
    YFinanceSource   - default, no API key, slow but bulletproof
    TwelveDataSource - faster, needs free API key, swap with one config line

The output contract is the same for both:
    {
        "dxy":    pd.DataFrame indexed by date with column "Close" (+ OHLC),
        "ief":    ...,
        "silver": ...,
        "sp500":  ...,
        "eurusd": ...,
        "vix":    ...,
        "gold":   ...   <- the TARGET instrument
    }
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import requests

try:
    from .config import Config
except ImportError:
    from config import Config


# -----------------------------------------------------------------------------
# Ticker maps  (Gold Scalper market set)
# -----------------------------------------------------------------------------
YFINANCE_TICKERS: Dict[str, str] = {
    "dxy":    "DX-Y.NYB",   # US Dollar Index
    "ief":    "IEF",        # iShares 7-10 Year Treasury Bond ETF (yield proxy)
    "silver": "SI=F",       # silver futures
    "sp500":  "SPY",        # S&P 500 ETF proxy
    "eurusd": "EURUSD=X",   # EUR / USD
    "vix":    "^VIX",       # CBOE VIX
    "gold":   "GC=F",       # gold futures  (the TARGET)
}

TWELVE_DATA_TICKERS: Dict[str, str] = {
    "dxy":    "DXY",
    "ief":    "IEF",
    "silver": "XAG/USD",
    "sp500":  "SPY",
    "eurusd": "EUR/USD",
    "vix":    "VIX",
    "gold":   "XAU/USD",
}


# -----------------------------------------------------------------------------
# OHLC normalization
# -----------------------------------------------------------------------------
def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the dataframe returned by yfinance / Twelve Data into a clean
    OHLCV form with Title-case columns and a tz-naive DatetimeIndex.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance sometimes returns multi-index columns when downloading
        # multiple tickers.  Drop the second level.
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna(how="all")


# -----------------------------------------------------------------------------
# Alignment helper (free function, used by both providers)
# -----------------------------------------------------------------------------
def _align(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Reindex every market onto the union of dates, forward-filling gaps
    (markets have different trading calendars).  Drops leading rows where
    GOLD (the master / target instrument) is missing.
    """
    if not data:
        return data
    if "gold" not in data:
        raise RuntimeError("Gold data missing — check your ticker / provider.")

    master_idx = data["gold"].index
    out = {"gold": data["gold"]}
    for mkt, df in data.items():
        if mkt == "gold":
            continue
        aligned = df.reindex(master_idx).ffill()
        out[mkt] = aligned
    return out


# -----------------------------------------------------------------------------
# Interval configuration
# -----------------------------------------------------------------------------
# yfinance enforces strict lookback limits per interval:
#   1m  -> 7 days, 5m/15m/30m/90m -> 60 days, 1h -> 730 days, 1d -> unlimited
# Twelve Data free tier caps each request at 5000 bars regardless.
INTERVAL_CONFIG: Dict[str, Dict] = {
    "1d":  {"lookback_days": 365 * 3, "bar_label": "day",   "bars_per_day": 1},
    "1h":  {"lookback_days": 60,      "bar_label": "hour",  "bars_per_day": 24},
    "30m": {"lookback_days": 30,      "bar_label": "30min", "bars_per_day": 32},
    "15m": {"lookback_days": 30,      "bar_label": "15min", "bars_per_day": 64},
    "5m":  {"lookback_days": 30,      "bar_label": "5min",  "bars_per_day": 192},
    "1m":  {"lookback_days": 5,       "bar_label": "1min",  "bars_per_day": 960},
}
SUPPORTED_INTERVALS = list(INTERVAL_CONFIG.keys())


def interval_lookback_days(interval: str) -> int:
    """Max lookback the provider supports for the given interval."""
    cfg = INTERVAL_CONFIG.get(interval)
    if cfg is None:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {SUPPORTED_INTERVALS}")
    return cfg["lookback_days"]


def interval_bar_label(interval: str) -> str:
    return INTERVAL_CONFIG.get(interval, {}).get("bar_label", interval)


# -----------------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------------
class DataSource:
    """Common interface for data providers."""

    name: str = "abstract"

    def fetch_all(
        self,
        lookback_days: int = 365 * 3,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# yfinance implementation  (default)
# -----------------------------------------------------------------------------
class YFinanceSource(DataSource):
    name = "yfinance"

    def fetch_all(self, lookback_days: int = 365 * 3, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        import yfinance as yf

        end = datetime.utcnow().date()
        start = end - timedelta(days=lookback_days)

        out: Dict[str, pd.DataFrame] = {}
        for mkt, ticker in YFINANCE_TICKERS.items():
            try:
                df = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
                df = _normalize_ohlc(df)
                if not df.empty:
                    out[mkt] = df
            except Exception as e:
                # Don't crash the whole app for one bad ticker
                print(f"[yfinance] {mkt} ({ticker}) failed: {e}")
        return _align(out)


# -----------------------------------------------------------------------------
# Twelve Data implementation  (swap target)
# -----------------------------------------------------------------------------
class TwelveDataSource(DataSource):
    """
    https://twelvedata.com — free tier: 800 requests/day, 8/min.
    Activate by setting `config.data_source = "twelvedata"` and
    `config.twelvedata_api_key = "..."`.
    """

    BASE = "https://api.twelvedata.com/time_series"
    name = "twelvedata"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Twelve Data API key is required (set in the sidebar).")
        self.api_key = api_key

    def _fetch_one(self, symbol: str, lookback_days: int, interval: str) -> pd.DataFrame:
        td_interval = {
            "1d":  "1day",
            "1h":  "1h",
            "30m": "30min",
            "15m": "15min",
            "5m":  "5min",
            "1m":  "1min",
        }.get(interval, interval)
        params = {
            "symbol": symbol,
            "interval": td_interval,
            "outputsize": max(80, lookback_days),
            "apikey": self.api_key,
            "format": "JSON",
            "order": "ASC",
        }
        r = requests.get(self.BASE, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        if "values" not in js:
            raise RuntimeError(f"Twelve Data error for {symbol}: {js.get('message', js)}")
        df = pd.DataFrame(js["values"])
        df = df.rename(columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def fetch_all(self, lookback_days: int = 365 * 3, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for mkt, symbol in TWELVE_DATA_TICKERS.items():
            for attempt in range(3):
                try:
                    df = self._fetch_one(symbol, lookback_days, interval)
                    df = _normalize_ohlc(df)
                    if not df.empty:
                        out[mkt] = df
                    break
                except Exception as e:
                    print(f"[twelvedata] {symbol} attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
        return _align(out)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
class DataSourceFactory:
    @staticmethod
    def create(config: Config) -> DataSource:
        if config.data_source == "yfinance":
            return YFinanceSource()
        if config.data_source == "twelvedata":
            return TwelveDataSource(config.twelvedata_api_key)
        raise ValueError(f"Unknown data source: {config.data_source}")


# -----------------------------------------------------------------------------
# Multi-timeframe fetch helper (used by the signal engine)
# -----------------------------------------------------------------------------
def fetch_multi_timeframe(
    config: Config,
    intervals: tuple = ("5m", "15m", "1h"),
    show_progress: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch the full 7-market data set on multiple timeframes at once.

    Returns a nested dict: {interval: {market_key: DataFrame}}

    Each provider call is independent (yfinance has hard per-interval
    lookback limits, so we can't just resample).  Use this for the
    multi-TF signal engine, not for the main UI.
    """
    source = DataSourceFactory.create(config)
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for interval in intervals:
        if interval not in INTERVAL_CONFIG:
            if show_progress:
                print(f"[fetch_multi] skipping unsupported interval {interval}")
            continue
        lookback = INTERVAL_CONFIG[interval]["lookback_days"]
        try:
            out[interval] = source.fetch_all(lookback_days=lookback, interval=interval)
        except Exception as e:
            if show_progress:
                print(f"[fetch_multi] {interval} failed: {e}")
            out[interval] = {}
    return out
