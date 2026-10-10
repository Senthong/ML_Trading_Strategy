"""Synthetic OHLCV generator so the pipeline runs offline (demo / tests)."""
import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n=2500, seed=0, start="2012-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    # mild momentum (AR(1)) + volatility clustering, so there is a *little* signal
    vol = 0.01 * np.exp(0.3 * np.cumsum(rng.normal(0, 0.05, n)) / np.sqrt(n) * 5)
    r = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        r[i] = 0.0004 + 0.05 * r[i - 1] + vol[i] * eps[i]
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
