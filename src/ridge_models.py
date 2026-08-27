"""Ridge regression paths and cross-validated regularization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ridge_pipeline(alpha=1.0):
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def ridge_cv(alphas=None):
    alphas = np.logspace(-4, 4, 60) if alphas is None else np.asarray(alphas, float)
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=alphas))])


def coefficient_path(x: pd.DataFrame, y, alphas=None) -> pd.DataFrame:
    alphas = np.logspace(-4, 4, 60) if alphas is None else np.asarray(alphas, float)
    rows=[]
    for alpha in alphas:
        model = ridge_pipeline(alpha).fit(x, y)
        coef = model.named_steps["ridge"].coef_
        rows.append({"alpha":float(alpha), **{f"coef_{name}":float(value) for name,value in zip(x.columns,coef)}})
    return pd.DataFrame(rows)
