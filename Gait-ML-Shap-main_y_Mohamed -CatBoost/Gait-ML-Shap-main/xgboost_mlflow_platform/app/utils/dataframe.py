from __future__ import annotations

from typing import Iterable

import pandas as pd


def first_existing(row: pd.Series, columns: Iterable[str], default=None):
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return default


def to_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def percent(value) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    return f"{number:.1%}"


def metric_range(df: pd.DataFrame, column: str) -> tuple[float, float]:
    if df.empty or column not in df.columns:
        return 0.0, 1.0
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return 0.0, 1.0
    low = max(0.0, float(series.min()))
    high = min(1.0, float(series.max()))
    if low == high:
        low = max(0.0, low - 0.01)
        high = min(1.0, high + 0.01)
    return low, high

