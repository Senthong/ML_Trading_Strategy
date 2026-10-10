"""Download, clean and cache OHLCV data (batch download + Parquet cache, scales to 500+ tickers)."""
import time
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
COLS = ["open", "high", "low", "close", "volume"]


def get_sp500_tickers() -> list[str]:
    """Current S&P 500 constituents (Wikipedia). NOTE: current list => survivorship bias."""
    import io, urllib.request
    req = urllib.request.Request("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                 headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    table = pd.read_html(io.StringIO(html))[0]
    return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())  # BRK.B -> BRK-B


def _split_batch(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split yf.download(group_by='ticker') output into one frame per ticker."""
    out = {}
    if not isinstance(raw.columns, pd.MultiIndex):          # single ticker case
        raw = pd.concat({tickers[0]: raw}, axis=1)
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        d = raw[t].copy()
        d.columns = [str(c).lower() for c in d.columns]
        if not set(COLS) <= set(d.columns):
            continue
        d = d[COLS].dropna(how="all")
        if len(d):
            d.index.name = "date"
            out[t] = d
    return out


def download_ohlcv(tickers, start, end=None, cache=True, chunk=50, pause=1.0) -> dict[str, pd.DataFrame]:
    """Batch-download adjusted daily OHLCV; cached per ticker as Parquet. Re-runs only fetch missing tickers."""
    import yfinance as yf

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tickers if not (cache and (RAW_DIR / f"{t}.parquet").exists())]
    for i in range(0, len(missing), chunk):
        batch = missing[i:i + chunk]
        print(f"  downloading {i + 1}-{i + len(batch)} / {len(missing)} ...")
        try:
            raw = yf.download(batch, start=start, end=end, auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
            for t, d in _split_batch(raw, batch).items():
                d.to_parquet(RAW_DIR / f"{t}.parquet")
        except Exception as e:                                # keep going, retry next run
            print(f"  batch failed ({e}); will be retried on next run")
        time.sleep(pause)                                     # be polite to the API

    out, failed = {}, []
    for t in tickers:
        f = RAW_DIR / f"{t}.parquet"
        if f.exists():
            out[t] = clean_ohlcv(pd.read_parquet(f))
        else:
            failed.append(t)
    if failed:
        print(f"  [warn] no data for {len(failed)} tickers: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    return out


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate, drop invalid rows, forward-fill tiny gaps."""
    df = df[COLS].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.ffill(limit=3).dropna()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["high"] >= df["low"]]
    return df.astype(float)
