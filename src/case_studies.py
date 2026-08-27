"""Synthetic banking and telecom case studies for regression diagnostics."""
from __future__ import annotations

import numpy as np

from regression_diagnostics import condition_number, cross_correlation, ols_fit, ridge_fit, vif


def banking_case(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    n = 800
    income = rng.lognormal(8.5, 0.45, n)
    debt = 0.34 * income + rng.normal(0, 2500, n)
    utilization = np.clip(debt / np.maximum(income, 1) + rng.normal(0, 0.05, n), 0, 1.5)
    balance = debt + rng.normal(0, 900, n)
    x = np.column_stack([income, debt, utilization, balance])
    y = 0.000015 * debt + 1.8 * utilization + rng.normal(0, 0.35, n)

    print("Banking multicollinearity")
    print("  condition number:", round(condition_number(x), 2))
    print("  VIF:", np.round(vif(x), 2))
    print("  OLS coefficients:", np.round(ols_fit(x, y), 5))
    print("  Ridge coefficients:", np.round(ridge_fit(x, y, alpha=50.0), 5))


def telecom_case(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n = 500
    congestion = rng.normal(0, 1, n)
    latency = np.roll(congestion, 4) + rng.normal(0, 0.4, n)
    lags, corr = cross_correlation(congestion, latency, max_lag=12)
    best = int(np.nanargmax(np.abs(corr)))
    print("\nTelecom lag analysis")
    print(f"  strongest lag: {lags[best]} samples")
    print(f"  correlation: {corr[best]:.3f}")


if __name__ == "__main__":
    banking_case()
    telecom_case()
