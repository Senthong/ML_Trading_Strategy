import numpy as np
from src.features import FEATURE_COLS, build_dataset
from src.synthetic import make_synthetic_ohlcv


def test_no_lookahead_in_features():
    """Changing FUTURE prices must not change features at earlier dates."""
    df = make_synthetic_ohlcv(n=400, seed=1)
    base = build_dataset(df)
    cut = base.index[200]
    df2 = df.copy()
    df2.loc[df2.index > cut, ["open", "high", "low", "close"]] *= 1.5   # perturb the future
    pert = build_dataset(df2)
    a = base.loc[:cut, FEATURE_COLS]; b = pert.loc[:cut, FEATURE_COLS]
    assert np.allclose(a.values, b.values, equal_nan=True)


def test_target_is_next_day_direction():
    df = make_synthetic_ohlcv(n=300, seed=2)
    ds = build_dataset(df)
    nxt = df["close"].pct_change().shift(-1).loc[ds.index]
    assert ((nxt > 0).astype(int) == ds["target"]).all()


def test_rsi_bounds():
    ds = build_dataset(make_synthetic_ohlcv(n=300, seed=3))
    assert ds["rsi_14"].between(0, 100).all()


def test_split_batch_multiindex():
    import pandas as pd
    from src.ingest import _split_batch
    idx = pd.date_range("2020-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw[("BBB", "Close")] = float("nan"); raw.loc[:, "BBB"] = float("nan")   # BBB has no data
    out = _split_batch(raw, ["AAA", "BBB", "CCC"])
    assert list(out) == ["AAA"] and list(out["AAA"].columns) == ["open", "high", "low", "close", "volume"]
