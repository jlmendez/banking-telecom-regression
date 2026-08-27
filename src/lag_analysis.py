"""Cross-correlation and lag scans for operational time-series relationships."""
from __future__ import annotations

import numpy as np
import pandas as pd


def lagged_correlation(x, y, max_lag: int = 24) -> pd.DataFrame:
    a = pd.Series(x, dtype=float); b = pd.Series(y, dtype=float)
    rows=[]
    for lag in range(-max_lag, max_lag+1):
        corr = a.corr(b.shift(lag))
        rows.append({"lag":lag,"correlation":float(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def best_lag(x, y, max_lag: int = 24):
    table = lagged_correlation(x, y, max_lag).dropna()
    row = table.loc[table.correlation.abs().idxmax()]
    return {"lag":int(row.lag),"correlation":float(row.correlation)}


def rolling_correlation(x, y, window: int = 30):
    return pd.Series(x, dtype=float).rolling(window).corr(pd.Series(y, dtype=float))
