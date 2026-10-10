"""Performance + classification metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

TRADING_DAYS = 252


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(np.sqrt(TRADING_DAYS) * r.mean() / sd) if sd > 0 else 0.0


def cagr(r: pd.Series) -> float:
    years = len(r) / TRADING_DAYS
    return float((1 + r).prod() ** (1 / years) - 1) if years > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def performance_summary(r: pd.Series, position: pd.Series | None = None) -> dict:
    out = {
        "CAGR": cagr(r),
        "Sharpe": sharpe(r),
        "MaxDD": max_drawdown(r),
        "AnnVol": float(r.std() * np.sqrt(TRADING_DAYS)),
    }
    if position is not None:
        active = r[position.reindex(r.index).fillna(0) > 0]
        out["WinRate(days in mkt)"] = float((active > 0).mean()) if len(active) else 0.0
        out["Exposure"] = float(position.mean())  # avg fraction of capital invested
    return out


def classification_summary(y_true, proba, threshold=0.5) -> dict:
    pred = (proba > threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else np.nan,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
    }
