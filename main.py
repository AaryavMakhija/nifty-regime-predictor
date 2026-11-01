#!/usr/bin/env python3
"""
Nifty 50 Market Regime & Volatility Predictor
=============================================

End-to-end pipeline:
    1. Download ~5 years of daily OHLCV for the Nifty 50 index (^NSEI) via yfinance.
    2. Clean the data (MultiIndex flattening, timezone/date parsing, NaN repair).
    3. Engineer technical features (RSI, MACD, Bollinger Bands, ATR%, EMAs,
       log returns, rolling volatility).
    4. Label forward-looking market regimes:
           0 = Bearish Trend
           1 = Bullish Trend
           2 = High Volatility / Consolidation
       using the 5-day forward log return and the 5-day forward average true range.
    5. Train an XGBoost classifier (falls back to RandomForest) on a strictly
       temporal 80/20 split with a purge gap to remove label-overlap leakage.
    6. Report Accuracy, Classification Report and Feature Importance.
    7. Predict the regime for the next trading session with class probabilities.
    8. Write an interactive, self-contained Plotly HTML report.

Usage:
    python main.py
    python main.py --years 7 --horizon 5 --model rf --output my_report.html

DISCLAIMER: Educational / research use only. Not investment advice.
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - environment guard
    sys.exit("Missing dependency 'yfinance'. Install with: pip install yfinance")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - environment guard
    sys.exit("Missing dependency 'plotly'. Install with: pip install plotly")

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost is optional. Catch *any* exception: on macOS a missing libomp raises
# XGBoostError rather than ImportError.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("nifty_regime")

REGIME_NAMES: Dict[int, str] = {
    0: "Bearish Trend",
    1: "Bullish Trend",
    2: "High Volatility / Consolidation",
}
REGIME_COLORS: Dict[int, str] = {
    0: "#dc2626",  # red
    1: "#16a34a",  # green
    2: "#f59e0b",  # amber
}

FEATURE_COLUMNS: List[str] = [
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "bb_bandwidth",
    "bb_pct_b",
    "atr_pct",
    "close_to_ema5",
    "close_to_ema20",
    "close_to_ema50",
    "ema5_to_ema20",
    "ema20_to_ema50",
    "log_ret",
    "ret_5d",
    "vol_10",
    "vol_30",
    "vol_ratio",
]


@dataclass
class Config:
    ticker: str = "^NSEI"
    years: int = 5
    horizon: int = 5                 # forward window (trading days) for labels
    train_frac: float = 0.80
    vol_quantile: float = 0.80       # trailing ATR% quantile => "high volatility"
    vol_lookback: int = 252          # trailing window for that quantile
    trend_k: float = 0.50            # |fwd_ret| must exceed k * sigma_5d to be a trend
    model: str = "auto"              # auto | xgb | rf
    seed: int = 42
    output: str = "nifty_regime_report.html"
    retries: int = 3


@dataclass
class ModelResult:
    model_name: str
    model: object
    accuracy: float
    baseline_accuracy: float
    report_text: str
    report_dict: Dict
    confusion: np.ndarray
    feature_importance: pd.Series
    test_index: pd.DatetimeIndex
    test_pred: np.ndarray
    test_true: np.ndarray
    n_train: int
    n_test: int
    purge: int
    extras: Dict = field(default_factory=dict)


@dataclass
class Prediction:
    as_of: pd.Timestamp
    target_session: pd.Timestamp
    regime_id: int
    regime_name: str
    probabilities: Dict[str, float]


# ---------------------------------------------------------------------------
# 1. Data pipeline
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's (Price, Ticker) MultiIndex columns into OHLCV names."""
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = [str(v) for v in df.columns.get_level_values(level)]
            if "Close" in values or "close" in values:
                df.columns = values
                break
        else:
            df.columns = ["_".join(map(str, c)) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    rename = {c: str(c).strip().title().replace("_", " ") for c in df.columns}
    rename = {k: ("Adj Close" if v.lower() == "adj close" else v) for k, v in rename.items()}
    return df.rename(columns=rename)


def clean_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns, parse dates, and repair/remove bad rows."""
    if raw is None or raw.empty:
        raise ValueError("Received an empty price frame.")

    df = _flatten_columns(raw.copy())

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[required + ["Volume"]]

    # --- Date parsing: datetime, tz-naive, normalised to the session date ---
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)  # keep exchange-local wall-clock date
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # --- Numeric coercion and NaN / inf handling ---
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df.loc[(df[required] <= 0).any(axis=1), required] = np.nan

    df = df.dropna(subset=["Close"])                 # a session without a close is unusable
    for col in ["Open", "High", "Low"]:              # repair isolated OHL gaps from the close
        df[col] = df[col].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    # Enforce OHLC consistency (vendor glitches occasionally violate it).
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)

    if len(df) < 300:
        raise ValueError(
            f"Only {len(df)} clean daily rows available; need >= 300 for a stable model."
        )
    return df


def fetch_data(cfg: Config) -> pd.DataFrame:
    """Download daily data for cfg.ticker over the last cfg.years years, with retries."""
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=cfg.years)
    LOGGER.info("Downloading %s daily data %s -> %s", cfg.ticker, start.date(), end.date())

    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.retries + 1):
        try:
            raw = yf.download(
                cfg.ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return clean_ohlcv(raw)
            raise ValueError("yf.download returned no rows")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning("Download attempt %d/%d failed: %s", attempt, cfg.retries, exc)
            time.sleep(2 * attempt)

    # Fallback: the Ticker.history endpoint sometimes succeeds when download() does not.
    try:
        LOGGER.info("Falling back to yf.Ticker(...).history()")
        raw = yf.Ticker(cfg.ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        return clean_ohlcv(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download data for {cfg.ticker}. Last error: {last_error or exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (an EMA with alpha = 1/period)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0.0), period)
    avg_loss = _wilder(-delta.clip(upper=0.0), period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 with gains => RSI 100; flat => 50
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi.where(avg_gain.notna())


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators. Every feature uses data up to and including day t."""
    out = df.copy()
    close = out["Close"]

    # RSI (14)
    out["rsi_14"] = compute_rsi(close, 14)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    # Price-normalised (in %) so the model is not fooled by the index level drifting.
    out["macd_norm"] = out["macd"] / close * 100.0
    out["macd_signal_norm"] = out["macd_signal"] / close * 100.0
    out["macd_hist_norm"] = out["macd_hist"] / close * 100.0

    # Bollinger Bands (20, 2)
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 2.0 * std
    out["bb_lower"] = mid - 2.0 * std
    band = (out["bb_upper"] - out["bb_lower"]).replace(0.0, np.nan)
    out["bb_bandwidth"] = band / mid
    out["bb_pct_b"] = (close - out["bb_lower"]) / band

    # ATR (14), normalised by close
    out["true_range"] = compute_true_range(out)
    out["atr_14"] = _wilder(out["true_range"], 14)
    out["atr_pct"] = out["atr_14"] / close

    # EMAs (5, 20, 50) and their relative distances
    for span in (5, 20, 50):
        out[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    out["close_to_ema5"] = close / out["ema_5"] - 1.0
    out["close_to_ema20"] = close / out["ema_20"] - 1.0
    out["close_to_ema50"] = close / out["ema_50"] - 1.0
    out["ema5_to_ema20"] = out["ema_5"] / out["ema_20"] - 1.0
    out["ema20_to_ema50"] = out["ema_20"] / out["ema_50"] - 1.0

    # Log returns and rolling (annualised) volatility
    out["log_ret"] = np.log(close / close.shift(1))
    out["ret_5d"] = np.log(close / close.shift(5))
    out["vol_10"] = out["log_ret"].rolling(10, min_periods=10).std() * np.sqrt(252)
    out["vol_30"] = out["log_ret"].rolling(30, min_periods=30).std() * np.sqrt(252)
    out["vol_ratio"] = out["vol_10"] / out["vol_30"].replace(0.0, np.nan)

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------------------
# 3. Target labelling
# ---------------------------------------------------------------------------

def label_regimes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Label the regime that *starts* with the next trading session.

    For each day t (h = cfg.horizon):
        fwd_ret      = ln(Close[t+h] / Close[t])
        fwd_atr_pct  = mean(TrueRange[t+1 .. t+h]) / Close[t]
        vol_thresh   = trailing quantile of atr_pct over the past year (known at t)
        trend_thresh = trend_k * daily_sigma_20 * sqrt(h)       (known at t)

        High Volatility / Consolidation (2) if fwd_atr_pct > vol_thresh
                                           or |fwd_ret| <= trend_thresh
        Bullish Trend (1)                  if fwd_ret > trend_thresh
        Bearish Trend (0)                  if fwd_ret < -trend_thresh

    Thresholds use only information available at t, so the label definition is
    consistent through time; the last h rows have no label and are excluded from training.
    """
    h = cfg.horizon
    out = df.copy()
    close, tr = out["Close"], out["true_range"]

    out["fwd_ret"] = np.log(close.shift(-h) / close)
    fwd_tr_mean = tr[::-1].rolling(h, min_periods=h).mean()[::-1].shift(-1)  # mean TR over t+1..t+h
    out["fwd_atr_pct"] = fwd_tr_mean / close

    out["vol_thresh"] = (
        out["atr_pct"]
        .rolling(cfg.vol_lookback, min_periods=60)
        .quantile(cfg.vol_quantile)
    )
    sigma_h = out["log_ret"].rolling(20, min_periods=20).std() * np.sqrt(h)
    out["trend_thresh"] = cfg.trend_k * sigma_h

    regime = np.where(out["fwd_ret"] > out["trend_thresh"], 1, 0)
    choppy = out["fwd_ret"].abs() <= out["trend_thresh"]
    high_vol = out["fwd_atr_pct"] > out["vol_thresh"]
    regime = np.where(choppy | high_vol, 2, regime)

    valid = out[["fwd_ret", "fwd_atr_pct", "vol_thresh", "trend_thresh"]].notna().all(axis=1)
    out["target"] = pd.Series(regime, index=out.index).where(valid).astype("Float64")
    return out


# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------

def build_model(cfg: Config, n_classes_present: int) -> Tuple[str, object]:
    use_xgb = cfg.model == "xgb" or (cfg.model == "auto" and XGBOOST_AVAILABLE)
    if use_xgb and not XGBOOST_AVAILABLE:
        LOGGER.warning("XGBoost requested but not importable; using RandomForest.")
        use_xgb = False
    if use_xgb and n_classes_present < 3:
        LOGGER.warning("Not all 3 regimes present in training data; using RandomForest.")
        use_xgb = False

    if use_xgb:
        return "XGBoost", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=cfg.seed,
            n_jobs=-1,
        )
    return "RandomForest", RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=cfg.seed,
        n_jobs=-1,
    )


def temporal_split(
    data: pd.DataFrame, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Chronological 80/20 split. The last `horizon` training rows are purged because
    their forward-looking labels overlap the first days of the test period.
    """
    n = len(data)
    split = int(n * cfg.train_frac)
    purge = cfg.horizon
    train = data.iloc[: max(split - purge, 0)]
    test = data.iloc[split:]
    if len(train) < 200 or len(test) < 30:
        raise ValueError(f"Not enough data after split (train={len(train)}, test={len(test)}).")
    return train, test, purge


def train_and_evaluate(df: pd.DataFrame, cfg: Config) -> Tuple[ModelResult, object, pd.DataFrame]:
    labeled = df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    labeled["target"] = labeled["target"].astype(int)
    train, test, purge = temporal_split(labeled, cfg)

    X_tr, y_tr = train[FEATURE_COLUMNS].to_numpy(), train["target"].to_numpy()
    X_te, y_te = test[FEATURE_COLUMNS].to_numpy(), test["target"].to_numpy()

    name, model = build_model(cfg, len(np.unique(y_tr)))
    LOGGER.info(
        "Training %s | train %s -> %s (%d rows) | purge %d | test %s -> %s (%d rows)",
        name, train.index[0].date(), train.index[-1].date(), len(train), purge,
        test.index[0].date(), test.index[-1].date(), len(test),
    )
    model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))

    y_pred = np.asarray(model.predict(X_te)).astype(int).ravel()
    labels = [0, 1, 2]
    names = [REGIME_NAMES[i] for i in labels]
    acc = accuracy_score(y_te, y_pred)

    majority = pd.Series(y_tr).mode().iloc[0]
    baseline = float(np.mean(y_te == majority))

    report_text = classification_report(
        y_te, y_pred, labels=labels, target_names=names, digits=3, zero_division=0
    )
    report_dict = classification_report(
        y_te, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_te, y_pred, labels=labels)

    # Refit on ALL labelled history for the live forecast.
    final_name, final_model = build_model(cfg, len(np.unique(labeled["target"])))
    y_all = labeled["target"].to_numpy()
    final_model.fit(
        labeled[FEATURE_COLUMNS].to_numpy(), y_all,
        sample_weight=compute_sample_weight("balanced", y_all),
    )

    importance = pd.Series(
        np.asarray(final_model.feature_importances_, dtype=float), index=FEATURE_COLUMNS
    )
    total = importance.sum()
    importance = (importance / total if total > 0 else importance).sort_values(ascending=False)

    result = ModelResult(
        model_name=final_name,
        model=model,
        accuracy=acc,
        baseline_accuracy=baseline,
        report_text=report_text,
        report_dict=report_dict,
        confusion=cm,
        feature_importance=importance,
        test_index=test.index,
        test_pred=y_pred,
        test_true=y_te,
        n_train=len(train),
        n_test=len(test),
        purge=purge,
        extras={"class_counts": labeled["target"].value_counts().sort_index().to_dict()},
    )
    return result, final_model, labeled


# ---------------------------------------------------------------------------
# 5. Next-session prediction
# ---------------------------------------------------------------------------

def predict_next_session(df: pd.DataFrame, model: object) -> Prediction:
    latest = df.dropna(subset=FEATURE_COLUMNS)
    if latest.empty:
        raise ValueError("No row with a complete feature set is available for prediction.")
    row = latest.iloc[[-1]]
    proba = np.asarray(model.predict_proba(row[FEATURE_COLUMNS].to_numpy()))[0]
    classes = [int(c) for c in getattr(model, "classes_", range(len(proba)))]

    probs = {REGIME_NAMES[i]: 0.0 for i in REGIME_NAMES}
    for cls, p in zip(classes, proba):
        probs[REGIME_NAMES[cls]] = float(p)
    best = max(probs, key=probs.get)
    best_id = next(k for k, v in REGIME_NAMES.items() if v == best)

    as_of = row.index[-1]
    # NSE trading holidays are not modelled; next weekday is used as the target session.
    target = as_of + pd.offsets.BDay(1)
    return Prediction(as_of, target, best_id, best, probs)


# ---------------------------------------------------------------------------
# 6. Interactive report
# ---------------------------------------------------------------------------

def _regime_spans(index: pd.DatetimeIndex, regimes: np.ndarray) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Collapse consecutive identical predictions into (start, end, regime) spans."""
    spans: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if len(index) == 0:
        return spans
    start, current = index[0], int(regimes[0])
    for i in range(1, len(index)):
        if int(regimes[i]) != current:
            spans.append((start, index[i], current))
            start, current = index[i], int(regimes[i])
    spans.append((start, index[-1] + pd.Timedelta(days=1), current))
    return spans


def build_figure(df: pd.DataFrame, result: ModelResult, pred: Prediction, cfg: Config) -> go.Figure:
    plot_df = df.dropna(subset=["bb_mid"])
    fi = result.feature_importance.sort_values(ascending=True)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.055,
        row_heights=[0.44, 0.14, 0.16, 0.26],
        subplot_titles=(
            "Panel 1 — Nifty 50 with Bollinger Bands & out-of-sample predicted regimes",
            "Panel 2a — RSI (14)",
            "Panel 2b — MACD (12, 26, 9)",
            f"Panel 3 — Feature importance ({result.model_name})",
        ),
    )

    # --- Panel 1: price + Bollinger Bands ---
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name="Nifty 50",
            increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"], name="BB Upper",
                             line=dict(color="#6366f1", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"], name="BB Lower",
                             line=dict(color="#6366f1", width=1), fill="tonexty",
                             fillcolor="rgba(99,102,241,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_mid"], name="BB Mid (SMA 20)",
                             line=dict(color="#6366f1", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema_50"], name="EMA 50",
                             line=dict(color="#0ea5e9", width=1.2)), row=1, col=1)

    # Out-of-sample regime shading on the test period
    for start, end, reg in _regime_spans(result.test_index, result.test_pred):
        fig.add_vrect(x0=start, x1=end, fillcolor=REGIME_COLORS[reg], opacity=0.13,
                      layer="below", line_width=0, row=1, col=1)
    for reg, name in REGIME_NAMES.items():  # legend swatches for the shading
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Predicted: {name}",
                                 marker=dict(symbol="square", size=12, color=REGIME_COLORS[reg])),
                      row=1, col=1)

    # Next-session forecast marker
    last_close = float(df.loc[pred.as_of, "Close"])
    fig.add_trace(
        go.Scatter(
            x=[pred.target_session], y=[last_close], mode="markers+text",
            marker=dict(symbol="star", size=18, color=REGIME_COLORS[pred.regime_id],
                        line=dict(color="#111827", width=1)),
            text=[f"Next: {pred.regime_name} ({pred.probabilities[pred.regime_name]:.0%})"],
            textposition="top left", name="Next-session forecast",
        ),
        row=1, col=1,
    )
    fig.add_vline(x=result.test_index[0], line=dict(color="#6b7280", dash="dash", width=1),
                  row=1, col=1)
    fig.add_annotation(x=result.test_index[0], y=float(plot_df["High"].max()),
                       text="Test period →", showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color="#6b7280"), row=1, col=1)

    # --- Panel 2a: RSI ---
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi_14"], name="RSI 14",
                             line=dict(color="#7c3aed", width=1.3)), row=2, col=1)
    for level, color in ((70, "#dc2626"), (30, "#16a34a")):
        fig.add_hline(y=level, line=dict(color=color, dash="dash", width=1), row=2, col=1)

    # --- Panel 2b: MACD ---
    hist_colors = np.where(plot_df["macd_hist"] >= 0, "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["macd_hist"], name="MACD Histogram",
                         marker_color=hist_colors, opacity=0.6), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd"], name="MACD Line",
                             line=dict(color="#2563eb", width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_signal"], name="Signal Line",
                             line=dict(color="#f97316", width=1.3)), row=3, col=1)

    # --- Panel 3: Feature importance ---
    fig.add_trace(
        go.Bar(x=fi.values, y=fi.index, orientation="h", name="Importance",
               marker_color="#334155", showlegend=False,
               hovertemplate="%{y}: %{x:.3f}<extra></extra>"),
        row=4, col=1,
    )

    # Axes: link the three time panels; panel 3 keeps its own axis.
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Index level", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_xaxes(title_text="Normalised importance", row=4, col=1)

    fig.update_layout(
        template="plotly_white",
        height=1450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=60, r=30, t=110, b=50),
        bargap=0.15,
    )
    return fig


def write_report(fig: go.Figure, result: ModelResult, pred: Prediction, cfg: Config) -> None:
    prob_rows = "".join(
        f"<tr><td><span class='dot' style='background:{REGIME_COLORS[i]}'></span>"
        f"{html.escape(REGIME_NAMES[i])}</td>"
        f"<td class='num'>{pred.probabilities[REGIME_NAMES[i]]:.1%}</td></tr>"
        for i in REGIME_NAMES
    )
    metric_rows = "".join(
        f"<tr><td>{html.escape(REGIME_NAMES[i])}</td>"
        f"<td class='num'>{result.report_dict[REGIME_NAMES[i]]['precision']:.3f}</td>"
        f"<td class='num'>{result.report_dict[REGIME_NAMES[i]]['recall']:.3f}</td>"
        f"<td class='num'>{result.report_dict[REGIME_NAMES[i]]['f1-score']:.3f}</td>"
        f"<td class='num'>{int(result.report_dict[REGIME_NAMES[i]]['support'])}</td></tr>"
        for i in REGIME_NAMES
    )
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True,
                            config={"responsive": True, "displaylogo": False})
    color = REGIME_COLORS[pred.regime_id]

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nifty 50 Regime Report</title>
<style>
  :root {{ --fg:#0f172a; --muted:#64748b; --card:#ffffff; --bg:#f1f5f9; --line:#e2e8f0; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--fg); }}
  header {{ padding:28px 32px 8px; }}
  h1 {{ margin:0 0 4px; font-size:24px; }}
  .sub {{ color:var(--muted); font-size:14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
          gap:16px; padding:16px 32px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 20px; }}
  .card h2 {{ margin:0 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:.06em;
             color:var(--muted); }}
  .big {{ font-size:22px; font-weight:700; color:{color}; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  td, th {{ padding:6px 4px; border-bottom:1px solid var(--line); text-align:left; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:8px; }}
  .chart {{ margin:0 32px 24px; background:var(--card); border:1px solid var(--line);
           border-radius:12px; padding:8px; }}
  footer {{ color:var(--muted); font-size:12px; padding:0 32px 32px; line-height:1.5; }}
  @media (max-width:640px) {{ header,.grid {{ padding-left:16px; padding-right:16px; }}
                              .chart {{ margin:0 16px 16px; }} footer {{ padding:0 16px 24px; }} }}
</style>
</head>
<body>
<header>
  <h1>Nifty 50 Market Regime &amp; Volatility Predictor</h1>
  <div class="sub">Ticker {html.escape(cfg.ticker)} · data through {pred.as_of:%d %b %Y} ·
  generated {pd.Timestamp.now():%d %b %Y %H:%M}</div>
</header>
<section class="grid">
  <div class="card">
    <h2>Next-session forecast ({pred.target_session:%a %d %b %Y})</h2>
    <div class="big">{html.escape(pred.regime_name)}</div>
    <table>{prob_rows}</table>
  </div>
  <div class="card">
    <h2>Out-of-sample performance</h2>
    <table>
      <tr><td>Model</td><td class="num">{html.escape(result.model_name)}</td></tr>
      <tr><td>Test accuracy</td><td class="num">{result.accuracy:.1%}</td></tr>
      <tr><td>Majority-class baseline</td><td class="num">{result.baseline_accuracy:.1%}</td></tr>
      <tr><td>Macro F1</td><td class="num">{result.report_dict['macro avg']['f1-score']:.3f}</td></tr>
      <tr><td>Train / purge / test rows</td>
          <td class="num">{result.n_train} / {result.purge} / {result.n_test}</td></tr>
    </table>
  </div>
  <div class="card">
    <h2>Per-regime metrics (test set)</h2>
    <table>
      <tr><th>Regime</th><th class="num">Prec.</th><th class="num">Recall</th>
          <th class="num">F1</th><th class="num">N</th></tr>
      {metric_rows}
    </table>
  </div>
</section>
<div class="chart">{plot_html}</div>
<footer>
  Regimes are defined on a {cfg.horizon}-day forward window: a trend requires the forward log
  return to exceed {cfg.trend_k}× the trailing 20-day σ scaled to the horizon; a period is flagged
  High Volatility / Consolidation when the forward average true range exceeds the trailing
  {cfg.vol_lookback}-day {int(cfg.vol_quantile * 100)}th percentile of ATR%, or when the move is
  too small to count as a trend. Shading in Panel 1 shows out-of-sample predictions only.
  The next-session date assumes the next weekday and does not account for NSE holidays.<br>
  <strong>Disclaimer:</strong> for educational and research purposes only. This is not investment
  advice. Past performance and model accuracy do not guarantee future results.
</footer>
</body>
</html>"""
    with open(cfg.output, "w", encoding="utf-8") as fh:
        fh.write(page)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="Nifty 50 Market Regime & Volatility Predictor")
    p.add_argument("--ticker", default="^NSEI", help="Yahoo Finance ticker (default: ^NSEI)")
    p.add_argument("--years", type=int, default=5, help="Years of history to download (default: 5)")
    p.add_argument("--horizon", type=int, default=5, help="Forward label window in sessions (default: 5)")
    p.add_argument("--model", choices=["auto", "xgb", "rf"], default="auto",
                   help="auto = XGBoost if installed, else RandomForest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="nifty_regime_report.html", help="HTML report path")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    if a.years < 2:
        p.error("--years must be at least 2")
    if not 1 <= a.horizon <= 20:
        p.error("--horizon must be between 1 and 20")
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return Config(ticker=a.ticker, years=a.years, horizon=a.horizon,
                  model=a.model, seed=a.seed, output=a.output)


def run(cfg: Config) -> Tuple[ModelResult, Prediction]:
    prices = fetch_data(cfg)
    LOGGER.info("Clean data: %d sessions (%s -> %s)",
                len(prices), prices.index[0].date(), prices.index[-1].date())

    features = engineer_features(prices)
    labeled_df = label_regimes(features, cfg)

    result, final_model, labeled = train_and_evaluate(labeled_df, cfg)
    pred = predict_next_session(labeled_df, final_model)

    bar = "=" * 72
    print(f"\n{bar}\n NIFTY 50 MARKET REGIME PREDICTOR — {result.model_name}\n{bar}")
    counts = {REGIME_NAMES[k]: v for k, v in result.extras["class_counts"].items()}
    print(f"Label distribution (all labelled rows): {counts}")
    print(f"\nTest accuracy           : {result.accuracy:.4f}")
    print(f"Majority-class baseline : {result.baseline_accuracy:.4f}")
    print("\nClassification report (out-of-sample):\n" + result.report_text)
    print("Confusion matrix (rows = true, cols = predicted; Bear, Bull, HV/Cons):")
    print(result.confusion)
    print("\nFeature importance (final model):")
    for name, val in result.feature_importance.items():
        print(f"  {name:<18} {val:7.4f}  {'#' * int(round(val * 100))}")
    print(f"\n{bar}\n FORECAST for next session {pred.target_session:%a %d %b %Y} "
          f"(data as of {pred.as_of:%d %b %Y})\n{bar}")
    print(f"  Predicted regime: {pred.regime_name}")
    for name, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<34} {p:6.1%}")

    fig = build_figure(labeled_df, result, pred, cfg)
    write_report(fig, result, pred, cfg)
    print(f"\nInteractive report saved to: {cfg.output}")
    print("Disclaimer: educational use only — not investment advice.\n")
    return result, pred


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        run(cfg)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
