# Nifty 50 Market Regime & Volatility Predictor

A single-file Python pipeline that downloads five years of Nifty 50 (`^NSEI`) daily data, engineers technical features, labels forward-looking market regimes, trains a gradient-boosted (or random-forest) classifier without lookahead bias, and forecasts the regime for the next trading session — delivered as an interactive HTML report.

## Project Overview

The model classifies each trading day into one of three regimes for the **coming 5-session window**:

| ID | Regime | Rule (applied to the 5-day forward window) |
|----|--------|--------------------------------------------|
| 0 | **Bearish Trend** | Forward log return < −0.5 × trailing σ (scaled to 5 days) |
| 1 | **Bullish Trend** | Forward log return > +0.5 × trailing σ (scaled to 5 days) |
| 2 | **High Volatility / Consolidation** | Forward average true range exceeds the trailing 1-year 80th percentile of ATR%, **or** the move is too small to qualify as a trend |

All thresholds use only information available on the labelling day, so the regime definition stays consistent through time.

## Features

- **Robust data pipeline** — `yfinance` download with retries and a `Ticker.history()` fallback; handles MultiIndex columns, timezone-aware dates, duplicate rows, NaN/inf values and inconsistent OHLC bars.
- **Technical indicators** (pure `pandas`/`numpy`):
  - RSI (14, Wilder smoothing)
  - MACD line, signal and histogram (12, 26, 9)
  - Bollinger Bands (20, 2σ): upper, lower, bandwidth, %B
  - ATR (14) normalised by close
  - EMA 5 / 20 / 50 and price-to-EMA distances
  - Log returns, 5-day return, 10- and 30-day annualised rolling volatility
- **Model** — XGBoost when installed, automatic fallback to `RandomForestClassifier`; balanced sample weights for uneven regime frequencies.
- **Leak-free evaluation** — chronological 80/20 split with a purge gap equal to the label horizon, so no training label overlaps the test period.
- **Metrics** — accuracy, majority-class baseline, full classification report, confusion matrix and feature importance.
- **Next-session forecast** — the model is refit on all labelled history and predicts the next session's regime with class probabilities.
- **Interactive report** (`nifty_regime_report.html`, fully self-contained, works offline):
  - Panel 1: candlestick price, Bollinger Bands, EMA 50, out-of-sample predicted-regime shading and the next-session forecast marker
  - Panel 2: RSI (with 70/30 levels) and MACD with histogram
  - Panel 3: feature-importance bar chart
  - Summary cards for the forecast, test accuracy and per-regime metrics

## Installation

Requires Python 3.9+.

```bash
pip install yfinance pandas numpy scikit-learn plotly
```

Optional, for the XGBoost model:

```bash
pip install xgboost
```

## Usage

```bash
python main.py
```

The script prints the evaluation metrics and forecast to the console and writes `nifty_regime_report.html` to the current folder — open it in any browser.

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `^NSEI` | Any Yahoo Finance ticker (e.g. `^NSEBANK`) |
| `--years` | `5` | Years of history to download |
| `--horizon` | `5` | Forward window (sessions) used for regime labels |
| `--model` | `auto` | `auto`, `xgb` or `rf` |
| `--seed` | `42` | Random seed |
| `--output` | `nifty_regime_report.html` | Report path |
| `-v` | off | Verbose logging |

Example:

```bash
python main.py --model rf --years 7 --output reports/nifty.html
```

## Notes & Limitations

- The "next session" date is the next weekday; NSE trading holidays are not modelled.
- Yahoo Finance reports zero volume for `^NSEI`, so volume is not used as a feature.
- Regime classification on daily index data is a noisy problem — always compare test accuracy with the printed majority-class baseline before reading anything into a forecast.

## Disclaimer

This project is for **educational and research purposes only**. It is not investment, trading or financial advice, and the author is not a SEBI-registered investment adviser. Model outputs are probabilistic and can be wrong; past performance and backtested accuracy do not guarantee future results. Data is sourced from Yahoo Finance via `yfinance` and may contain errors or delays. Use at your own risk.
