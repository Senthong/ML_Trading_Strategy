"""Optional: persist raw / processed data in PostgreSQL."""
import os
import pandas as pd


def get_engine(url: str | None = None):
    from sqlalchemy import create_engine
    return create_engine(url or os.environ["DATABASE_URL"])  # postgresql+psycopg2://user:pw@host/db


def save_ohlcv(data: dict[str, pd.DataFrame], engine, table="ohlcv_daily") -> None:
    frames = [d.reset_index().assign(ticker=t) for t, d in data.items()]
    pd.concat(frames).rename(columns={"index": "date"}).to_sql(
        table, engine, if_exists="replace", index=False, method="multi", chunksize=5000
    )


def save_features(df: pd.DataFrame, engine, table="features_daily") -> None:
    df.reset_index().to_sql(table, engine, if_exists="replace", index=False, method="multi", chunksize=5000)
