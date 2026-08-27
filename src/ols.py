"""Matrix-form ordinary least squares and projection diagnostics."""
from __future__ import annotations

import numpy as np


def add_intercept(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(len(x)), x])


def fit_ols(x, y):
    X = add_intercept(x)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residual = y - fitted
    return beta, fitted, residual


def hat_matrix(x):
    X = add_intercept(x)
    return X @ np.linalg.pinv(X.T @ X) @ X.T


def r_squared(y, fitted):
    y = np.asarray(y, float); fitted = np.asarray(fitted, float)
    return float(1 - np.sum((y-fitted)**2) / np.sum((y-y.mean())**2))
