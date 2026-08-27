"""Variance inflation, singular values and conditioning diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.astype(float)
    rows = []
    for column in x.columns:
        others = x.drop(columns=column)
        r2 = LinearRegression().fit(others, x[column]).score(others, x[column]) if others.shape[1] else 0.0
        rows.append({"feature": column, "vif": float(1 / max(1-r2, 1e-12))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def svd_diagnostics(x):
    matrix = np.asarray(x, float)
    centered = matrix - matrix.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    condition = float(singular_values.max() / max(singular_values.min(), 1e-12))
    return {"singular_values": singular_values, "condition_number": condition}


def near_linear_dependencies(x, threshold_ratio=1e-3):
    matrix = np.asarray(x, float)
    centered = matrix - matrix.mean(axis=0)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    mask = s / max(s.max(), 1e-12) < threshold_ratio
    return vh[mask]
