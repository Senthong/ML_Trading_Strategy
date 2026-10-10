# ML-Based Quantitative Trading Strategy

Dự báo xác suất tăng/giảm của phiên kế tiếp từ dữ liệu OHLCV, rồi kiểm tra tín hiệu có tạo ra chiến lược khả thi sau phí giao dịch hay không.

**Stack:** Python · pandas · scikit-learn · XGBoost · yfinance · (PostgreSQL tùy chọn)

## Pipeline
```
yfinance -> clean OHLCV -> features (returns, SMA, RSI, volatility) -> target (next-day up/down)
   -> walk-forward training (LogReg / RandomForest / XGBoost) -> P(up)
   -> long/flat signal -> vectorized backtest (fees + slippage)
   -> compare vs Buy & Hold and SMA 20/50 crossover
```

## Cấu trúc
```
config.py            # tickers, phí, threshold, tham số walk-forward
run.py               # chạy toàn bộ pipeline
src/ingest.py        # tải + làm sạch OHLCV (cache CSV)
src/features.py      # feature + target (không look-ahead)
src/models.py        # models + walk-forward expanding window
src/backtest.py      # backtest vectorized có phí
src/metrics.py       # Sharpe, CAGR, MaxDD, AUC...
src/db.py, sql/      # lưu dữ liệu vào PostgreSQL (tùy chọn)
tests/               # unit test (no look-ahead, tính phí, drawdown)
```

## Chạy
```bash
pip install -r requirements.txt
python run.py --universe sp500 --jobs -1 --fast          # ~500 mã, dùng mọi CPU core
python run.py --universe sp500 --models LogReg XGBoost   # bỏ RandomForest cho nhanh
python run.py --universe sp500 --n-tickers 100 --start 2005-01-01
DATABASE_URL=postgresql+psycopg2://user:pw@localhost/db python run.py --to-postgres
```
Kết quả: `reports/performance.csv`, `reports/classification.csv`, `reports/equity_curve.png`.

## Các quyết định thiết kế quan trọng
- **Không look-ahead:** feature tại ngày *t* chỉ dùng dữ liệu đến close *t*; target = hướng của return *t→t+1* (`shift(-1)`); có test kiểm tra.
- **Walk-forward validation** (expanding window, re-fit mỗi 63 ngày) thay vì `train_test_split` ngẫu nhiên, vì dữ liệu chuỗi thời gian.
- **Phí + slippage** tính trên mỗi lần đổi vị thế (turnover).
- **Benchmark** đánh giá trên cùng khoảng out-of-sample với các mô hình ML.

## Kết quả
> Điền sau khi chạy với dữ liệu thật (đừng dùng số từ chế độ `--synthetic`).

| Strategy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| XGBoost | | | |
| Buy & Hold | | | |

## Hạn chế
- Dự báo hướng giá ngày kế tiếp có tín hiệu rất yếu; AUC gần 0.5 là bình thường.
- Chưa tính thuế, thanh khoản, short-selling; có survivorship bias (chỉ chọn mã đang tồn tại).
- Nên thử: nhiều mã hơn, tuning trong inner walk-forward, position sizing, đa lớp tín hiệu.

## Chạy với dữ liệu lớn (S&P 500, 15+ năm)
```bash
python run.py --universe sp500 --jobs -1 --fast                  # tất cả CPU core, mô hình nhẹ hơn
python run.py --universe sp500 --models LogReg XGBoost           # bỏ RandomForest (chậm nhất)
python run.py --universe sp500 --n-tickers 100 --start 2005-01-01
python run.py --universe sp500 --cost-bps 10                     # đổi phí: dùng lại checkpoint, chạy vài giây
```
- Tải dữ liệu theo batch 50 mã, cache Parquet ở `data/raw/` (chạy lại chỉ tải mã còn thiếu).
- Mỗi mã chạy trên 1 process (joblib); dự báo được checkpoint ở `data/checkpoints/` nên bị ngắt vẫn chạy tiếp được. Dùng `--fresh` để train lại.
- Thời gian đo trên 1 mã × 4000 ngày, 1 core: LogReg ~0.4s, XGBoost ~6s, RandomForest ~37s. Ước lượng tổng = số mã × thời gian ÷ số core.
- Lưu ý: danh sách S&P 500 hiện tại gây **survivorship bias** (bỏ các mã đã bị loại), cần nêu rõ khi trình bày kết quả.
