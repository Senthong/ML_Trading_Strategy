-- Optional PostgreSQL schema (pandas.to_sql creates tables automatically, this is for reference/production).
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    ticker  TEXT        NOT NULL,
    date    DATE        NOT NULL,
    open    NUMERIC(14,4), high NUMERIC(14,4), low NUMERIC(14,4), close NUMERIC(14,4),
    volume  BIGINT,
    PRIMARY KEY (ticker, date)
);

-- Example: daily return per ticker
-- SELECT ticker, date, close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
-- FROM ohlcv_daily;
