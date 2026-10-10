"""Models + walk-forward (expanding window) prediction."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def get_models(seed: int = 42, n_jobs: int = 1, fast: bool = False) -> dict:
    """n_jobs=1 when tickers are already parallelised across processes. fast=True trades accuracy for speed."""
    rf_trees, xgb_trees = (100, 150) if fast else (200, 200)
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_trees, max_depth=5, min_samples_leaf=20, max_samples=0.5 if fast else None,
            n_jobs=n_jobs, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgb_trees, max_depth=3, learning_rate=0.05, subsample=0.8, tree_method="hist",
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=n_jobs, random_state=seed,
        ),
    }


def walk_forward_predict(X, y, model, min_train=504, retrain_every=63, gap=1) -> pd.Series:
    """Out-of-sample P(up). Expanding window, re-fit every `retrain_every` days.

    A prediction at block start `s` is made by a model trained only on rows < s - gap,
    so no future information leaks into training.
    """
    n = len(X)
    probs = pd.Series(np.nan, index=X.index, name="proba")
    for start in range(min_train, n, retrain_every):
        end = min(start + retrain_every, n)
        m = clone(model)
        m.fit(X.iloc[: start - gap], y.iloc[: start - gap])
        probs.iloc[start:end] = m.predict_proba(X.iloc[start:end])[:, 1]
    return probs
