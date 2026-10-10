"""Per-ticker heavy work (features + walk-forward predictions), with on-disk checkpoints."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, build_dataset
from src.models import get_models, walk_forward_predict

CKPT_DIR = Path("data/checkpoints")


def run_tag(model_names, min_train, retrain_every, fast) -> str:
    key = f"{sorted(model_names)}|{min_train}|{retrain_every}|{fast}|{FEATURE_COLS}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def process_ticker(ticker, df, model_names, min_train, retrain_every, fast, seed=42, fresh=False):
    """Return OOS frame [target, fwd_ret, sma_signal, proba_<model>...] or None if too little data.

    Predictions (the expensive part) are checkpointed; backtest params (costs, threshold) can be
    changed later without retraining.
    """
    tag = run_tag(model_names, min_train, retrain_every, fast)
    ckpt = CKPT_DIR / tag / f"{ticker}.parquet"
    if ckpt.exists() and not fresh:
        return pd.read_parquet(ckpt)

    ds = build_dataset(df)
    if len(ds) < min_train + retrain_every:
        return None
    X = ds[FEATURE_COLS].astype(np.float32)
    y = ds["target"]
    models = get_models(seed, n_jobs=1, fast=fast)
    oos_idx = ds.index[min_train:]
    out = ds.loc[oos_idx, ["target", "fwd_ret", "sma_signal"]].copy()
    for name in model_names:
        out[f"proba_{name}"] = walk_forward_predict(X, y, models[name], min_train, retrain_every).loc[oos_idx]

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ckpt)
    return out
