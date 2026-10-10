"""End-to-end pipeline, built for large universes (parallel + checkpointed).

Examples:
    python run.py --synthetic --n-tickers 20               # offline demo
    python run.py --universe sp500 --jobs -1 --fast        # ~500 tickers, all CPU cores
    python run.py --universe sp500 --models LogReg XGBoost # skip the slow RandomForest
    python run.py --universe sp500 --cost-bps 10           # change costs: reuses checkpoints, no retraining
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import config as cfg
from src.backtest import backtest
from src.ingest import clean_ohlcv, download_ohlcv, get_sp500_tickers
from src.metrics import classification_summary, performance_summary
from src.pipeline import process_ticker
from src.synthetic import make_synthetic_ohlcv

ALL_MODELS = ["LogReg", "RandomForest", "XGBoost"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["default", "sp500"], default="default")
    p.add_argument("--n-tickers", type=int, default=None, help="limit number of tickers")
    p.add_argument("--start", default=cfg.START)
    p.add_argument("--synthetic", action="store_true", help="synthetic data (offline)")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    p.add_argument("--jobs", type=int, default=-1, help="parallel processes (-1 = all cores)")
    p.add_argument("--fast", action="store_true", help="lighter models (fewer trees) for big runs")
    p.add_argument("--retrain-every", type=int, default=cfg.RETRAIN_EVERY)
    p.add_argument("--min-train", type=int, default=cfg.MIN_TRAIN)
    p.add_argument("--cost-bps", type=float, default=cfg.COST_BPS)
    p.add_argument("--slippage-bps", type=float, default=cfg.SLIPPAGE_BPS)
    p.add_argument("--threshold", type=float, default=cfg.THRESHOLD)
    p.add_argument("--fresh", action="store_true", help="ignore checkpoints and retrain")
    p.add_argument("--to-postgres", action="store_true")
    return p.parse_args()


def load_data(a) -> dict[str, pd.DataFrame]:
    if a.synthetic:
        n = a.n_tickers or 3
        return {f"SYN{i:03d}": clean_ohlcv(make_synthetic_ohlcv(seed=i)) for i in range(n)}
    tickers = get_sp500_tickers() if a.universe == "sp500" else cfg.TICKERS
    if a.n_tickers:
        tickers = tickers[: a.n_tickers]
    print(f"Loading {len(tickers)} tickers ...")
    return download_ohlcv(tickers, a.start, cfg.END)


def main(a):
    data = load_data(a)
    if a.to_postgres:
        from src.db import get_engine, save_ohlcv
        save_ohlcv(data, get_engine())

    print(f"Training {a.models} on {len(data)} tickers (jobs={a.jobs}, fast={a.fast}) ...")
    results = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(process_ticker)(t, df, a.models, a.min_train, a.retrain_every, a.fast, cfg.RANDOM_STATE, a.fresh)
        for t, df in data.items()
    )
    frames = {t: r for t, r in zip(data, results) if r is not None}
    print(f"{len(frames)} tickers used ({len(data) - len(frames)} skipped: too little history)")
    del data

    # ---- cheap part: backtests (re-run instantly when costs/threshold change) ----
    strat_rets, strat_pos = {}, {}
    clf_y = {m: [] for m in a.models}; clf_p = {m: [] for m in a.models}
    for t, f in frames.items():
        signals = {f"ML: {m}": (f[f"proba_{m}"] > a.threshold).astype(int) for m in a.models}
        signals["SMA 20/50 crossover"] = f["sma_signal"]
        signals["Buy & Hold"] = pd.Series(1, index=f.index)
        for name, pos in signals.items():
            bt = backtest(f["fwd_ret"], pos, a.cost_bps, a.slippage_bps)
            strat_rets.setdefault(name, {})[t] = bt["net_ret"]
            strat_pos.setdefault(name, {})[t] = bt["position"]
        for m in a.models:
            clf_y[m].append(f["target"]); clf_p[m].append(f[f"proba_{m}"])

    port_ret = {s: pd.DataFrame(d).mean(axis=1) for s, d in strat_rets.items()}   # equal-weight
    port_pos = {s: pd.DataFrame(d).mean(axis=1) for s, d in strat_pos.items()}
    perf = pd.DataFrame({s: performance_summary(r, port_pos[s]) for s, r in port_ret.items()}).T
    clf = pd.DataFrame({m: classification_summary(pd.concat(clf_y[m]), pd.concat(clf_p[m]), a.threshold)
                        for m in a.models}).T

    # per-ticker view: how often does each ML model beat Buy & Hold on Sharpe?
    from src.metrics import sharpe
    bh = {t: sharpe(r) for t, r in strat_rets["Buy & Hold"].items()}
    beat = {s: sum(sharpe(r) > bh[t] for t, r in d.items()) / len(d) for s, d in strat_rets.items() if s.startswith("ML")}

    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 140
    print("\n=== OOS classification (pooled) ===\n", clf)
    print(f"\n=== Equal-weight portfolio, {len(frames)} tickers (cost={a.cost_bps}bps, slippage={a.slippage_bps}bps) ===\n", perf)
    print("\n=== % of tickers where ML Sharpe > Buy & Hold Sharpe ===")
    for s, v in beat.items():
        print(f"  {s:<20} {v:.1%}")

    out = Path("reports"); out.mkdir(exist_ok=True)
    perf.to_csv(out / "performance.csv"); clf.to_csv(out / "classification.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for s, r in port_ret.items():
        (1 + r).cumprod().plot(ax=ax, label=s, lw=2 if s == "Buy & Hold" else 1.2)
    ax.set_title(f"Equity curve, equal-weight {len(frames)} tickers (OOS, net of costs)")
    ax.set_ylabel("Growth of 1"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "equity_curve.png", dpi=150)
    print(f"\nSaved reports/ (performance.csv, classification.csv, equity_curve.png)")


if __name__ == "__main__":
    main(parse_args())
