"""Regression diagnostics based on matrix algebra and numerical conditioning."""
from __future__ import annotations

import numpy as np


def add_intercept(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.ones(len(x)), x])


def ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stable least-squares solution using NumPy's SVD-backed solver."""
    design = add_intercept(x)
    beta, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
    return beta


def condition_number(x: np.ndarray, standardize: bool = True) -> float:
    x = np.asarray(x, dtype=float)
    if standardize:
        scale = x.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        x = (x - x.mean(axis=0)) / scale
    return float(np.linalg.cond(x))


def singular_values(x: np.ndarray, standardize: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if standardize:
        scale = x.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        x = (x - x.mean(axis=0)) / scale
    return np.linalg.svd(x, compute_uv=False)


def vif(x: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor for each feature."""
    x = np.asarray(x, dtype=float)
    result = []
    for j in range(x.shape[1]):
        target = x[:, j]
        others = np.delete(x, j, axis=1)
        pred = add_intercept(others) @ ols_fit(others, target)
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        result.append(float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2))
    return np.asarray(result)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form Ridge regression with an unpenalized intercept."""
    design = add_intercept(x)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + alpha * penalty, design.T @ np.asarray(y, dtype=float))


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Pearson cross-correlation across symmetric integer lags."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.arange(-max_lag, max_lag + 1)
    values = []
    for lag in lags:
        if lag < 0:
            xa, ya = x[-lag:], y[:lag]
        elif lag > 0:
            xa, ya = x[:-lag], y[lag:]
        else:
            xa, ya = x, y
        values.append(np.corrcoef(xa, ya)[0, 1])
    return lags, np.asarray(values)
