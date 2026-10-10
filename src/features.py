"""Feature engineering + target. Row t only uses information available at close of day t."""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "sma10_ratio", "sma20_ratio", "sma_cross_ratio",
    "rsi_14", "vol_10", "vol_20", "vol_ratio",
    "hl_range", "volume_chg",
]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(avg_gain.notna())


def build_dataset(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Return features + fwd_ret + target + sma_signal for one ticker.

    fwd_ret[t]  = close[t+1]/close[t] - 1   (the return earned by a position taken at close t)
    target[t]   = 1 if fwd_ret[t] > 0
    The last row has no fwd_ret and is dropped.
    """
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    ret = c.pct_change()

    for n in (1, 5, 10, 20):
        out[f"ret_{n}"] = c.pct_change(n)

    sma10, sma20, sma_f, sma_s = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(fast).mean(), c.rolling(slow).mean()
    out["sma10_ratio"] = c / sma10 - 1
    out["sma20_ratio"] = c / sma20 - 1
    out["sma_cross_ratio"] = sma_f / sma_s - 1

    out["rsi_14"] = rsi(c, 14)
    out["vol_10"] = ret.rolling(10).std()
    out["vol_20"] = ret.rolling(20).std()
    out["vol_ratio"] = out["vol_10"] / out["vol_20"]
    out["hl_range"] = (df["high"] - df["low"]) / c
    out["volume_chg"] = df["volume"].pct_change().clip(-1, 5)

    # --- target (uses the FUTURE on purpose; never used as a feature) ---
    out["fwd_ret"] = c.pct_change().shift(-1)
    out["target"] = (out["fwd_ret"] > 0).astype(int)
    # baseline strategy signal, known at close t
    out["sma_signal"] = (sma_f > sma_s).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=FEATURE_COLS + ["fwd_ret"])
