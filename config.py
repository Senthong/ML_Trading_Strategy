"""Central configuration for the project."""
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
START = "2010-01-01"
END = None  # None = today

# Backtest
COST_BPS = 5.0        # commission per unit of turnover (basis points)
SLIPPAGE_BPS = 2.0    # slippage per unit of turnover (basis points)
THRESHOLD = 0.5       # P(up) above this -> long, else flat

# Walk-forward validation
MIN_TRAIN = 504       # ~2 years of trading days before first prediction
RETRAIN_EVERY = 63    # re-fit every ~quarter (expanding window)

RANDOM_STATE = 42
