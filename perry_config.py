from __future__ import annotations

import os

_PRIMARY_TIMEFRAME = os.environ.get("PERRY_TIMEFRAME", "15m")

SUPPORTED_TIMEFRAMES = [
    "1m", "2m", "3m", "5m", "7m", "10m", "12m", "15m", "30m", "45m",
    "1h", "2h", "4h", "1d", "3d", "7d",
]

SUPPORT_TIMEFRAME_MAP = {
    "5m": ["2m", "5m", "15m"],
    "1h": ["15m", "1h", "4h"],
    "1d": ["4h", "1d", "7d"],
}

BINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "2m": "2m",
    "3m": "3m",
    "5m": "5m",
    "7m": "7m",
    "10m": "10m",
    "12m": "12m",
    "15m": "15m",
    "30m": "30m",
    "45m": "45m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "3d": "3d",
    "7d": "7d",
}

PANDAS_OFFSET_MAP = {
    "1m": "1min",
    "2m": "2min",
    "3m": "3min",
    "5m": "5min",
    "7m": "7min",
    "10m": "10min",
    "12m": "12min",
    "15m": "15min",
    "30m": "30min",
    "45m": "45min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "3d": "3d",
    "7d": "7d",
}


def set_primary_timeframe(timeframe: str) -> None:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    global _PRIMARY_TIMEFRAME
    _PRIMARY_TIMEFRAME = timeframe
    os.environ["PERRY_TIMEFRAME"] = timeframe


def get_primary_timeframe(default: str | None = None) -> str:
    if default is not None:
        return default
    return _PRIMARY_TIMEFRAME


def validate_timeframe(tf: str) -> bool:
    return tf in SUPPORTED_TIMEFRAMES


def get_support_timeframes(primary: str | None = None) -> list[str]:
    primary = primary or get_primary_timeframe()
    return SUPPORT_TIMEFRAME_MAP.get(primary, [primary])


def filename_with_timeframe(base: str, timeframe: str | None = None, ext: str = "csv") -> str:
    timeframe = timeframe or get_primary_timeframe()
    return f"{base}_{timeframe}.{ext}"


def timeframe_to_seconds(timeframe: str | None = None) -> int:
    timeframe = timeframe or get_primary_timeframe()
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 3600
    if timeframe.endswith("d"):
        return int(timeframe[:-1]) * 86400
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def timeframe_to_pandas_offset(timeframe: str | None = None) -> str:
    timeframe = timeframe or get_primary_timeframe()
    return PANDAS_OFFSET_MAP.get(timeframe, get_primary_timeframe())


def binance_interval(timeframe: str | None = None) -> str:
    timeframe = timeframe or get_primary_timeframe()
    if timeframe not in BINANCE_INTERVAL_MAP:
        raise ValueError(f"Unsupported Binance interval for timeframe: {timeframe}")
    return BINANCE_INTERVAL_MAP[timeframe]


# Duplicate stale helpers removed. Runtime helpers above already support selected timeframe.