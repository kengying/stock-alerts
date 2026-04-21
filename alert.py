"""
Sudden Spike Swing-Trade Scanner
(for quick momentum swings on volume/price explosions)

Scans your watchlist (fast — runs in seconds)
Shows only SUDDEN spikes (vol ≥ 5× + price move ≥ 7%)
Gives clear BUY or SELL verdict with entry/stop/targets
Optimized for GitHub Actions alert view
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import smtplib
import argparse
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from zoneinfo import ZoneInfo

# ── Default watchlist (fast scanning — no full market) ────────────────────────
DEFAULT_WATCHLIST = [
    "BYND", "GME", "AMC", "MARA", "RIVN", "LCID", "SOFI", "PLTR", "NIO",
    "TSLA", "NVDA", "COIN", "HOOD", "SPCE", "BBAI", "WKHS", "CLOV",
    "SNDL", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "AMD", "INTC",
]

ET = ZoneInfo("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR LIBRARY (unchanged — pure pandas/numpy)
# ══════════════════════════════════════════════════════════════════════════════

def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else 50.0


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9) -> dict:
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return {
        "macd":      round(float(macd_line.iloc[-1]),   4),
        "signal":    round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]),   4),
        "bullish_cross": bool(macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]),
        "bearish_cross": bool(macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]),
        "above_zero": bool(macd_line.iloc[-1] > 0),
    }


def calc_bollinger(close: pd.Series, period=20, std_dev=2) -> dict:
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    price = close.iloc[-1]
    band_width = upper.iloc[-1] - lower.iloc[-1]
    bw    = band_width / sma.iloc[-1] * 100 if sma.iloc[-1] != 0 else 0
    pct_b = (price - lower.iloc[-1]) / band_width if band_width != 0 else 0.5
    return {
        "upper":         round(float(upper.iloc[-1]), 2),
        "middle":        round(float(sma.iloc[-1]),   2),
        "lower":         round(float(lower.iloc[-1]), 2),
        "pct_b":         round(float(pct_b), 3),
        "bandwidth":     round(float(bw), 2),
        "squeeze":       bool(bw < 5),
        "breakout_up":   bool(price > upper.iloc[-1]),
        "breakout_down": bool(price < lower.iloc[-1]),
    }


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return round(float(atr.iloc[-1]), 4) if not atr.empty else 0.0


def calc_ema(close: pd.Series, period: int) -> float:
    ema = close.ewm(span=period, adjust=False).mean()
    return round(float(ema.iloc[-1]), 4) if not ema.empty else float(close.iloc[-1])


def calc_vwap(df: pd.DataFrame) -> float:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    if cum_vol.iloc[-1] == 0:
        return float(df["Close"].iloc[-1])
    vwap = (typical * df["Volume"]).cumsum() / cum_vol
    return round(float(vwap.iloc[-1]), 4)


def calc_support_resistance(df: pd.DataFrame, n_pivots=3) -> dict:
    highs = df["High"].rolling(5, center=True).max()
    lows  = df["Low"].rolling(5,  center=True).min()
    resistance_levels = sorted(df["High"][df["High"] == highs].dropna().unique(), reverse=True)[:n_pivots]
    support_levels = sorted(df["Low"][df["Low"] == lows].dropna().unique())[:n_pivots]
    price       = float(df["Close"].iloc[-1])
    nearest_res = min(resistance_levels, key=lambda x: abs(x - price)) if resistance_levels else price * 1.05
    nearest_sup = min(support_levels,    key=lambda x: abs(x - price)) if support_levels else price * 0.95
    upside      = nearest_res - price
    downside    = max(price - nearest_sup, 0.01)
    return {
        "nearest_resistance": round(nearest_res, 2),
        "nearest_support":    round(nearest_sup, 2),
        "risk_reward":        round(upside / downside, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SWING SCORING ENGINE (tuned for sudden spikes)
# ══════════════════════════════════════════════════════════════════════════════

def swing_score(data: dict) -> dict:
    ind          = data.get("indicators", {})
    score        = 0
    reasons      = []

    rsi          = ind.get("rsi", 50)
    macd         = ind.get("macd", {})
    bb           = ind.get("bollinger", {})
    ema9         = ind.get("ema9",  0)
    ema21        = ind.get("ema21", 0)
    price        = data.get("session_close", 0)
    vwap         = ind.get("vwap", price)
    atr          = ind.get("atr",  0)
    volume_spike = data.get("volume_spike_x", 1)
    price_chg    = data.get("price_change_pct", 0)
    gap          = data.get("gap_from_prev_close_pct", 0)
    sr           = ind.get("support_resistance", {})
    rr           = sr.get("risk_reward", 0)

    # Heavy weight on sudden volume + price explosion
    if volume_spike >= 10:
        score += 4
        reasons.append(f"🔥 Volume {volume_spike}× — massive sudden interest")
    elif volume_spike >= 5:
        score += 3
        reasons.append(f"🚀 Volume {volume_spike}× — strong spike")

    if price_chg >= 10:
        score += 3
        reasons.append(f"💥 Price +{price_chg:.1f}% — explosive move")
    elif price_chg >= 7:
        score += 2

    if gap >= 5:
        score += 2
        reasons.append(f"Gap up {gap:+.1f}% — continuation candidate")

    # Momentum filters
    if rsi < 35:
        score += 2
        reasons.append(f"RSI {rsi:.0f} — oversold bounce setup")
    elif rsi > 75:
        score -= 3
        reasons.append(f"RSI {rsi:.0f} — overbought, fade risk")

    if macd.get("bullish_cross"):
        score += 3
        reasons.append("MACD bullish cross — momentum flipping up")
    elif macd.get("above_zero") and macd.get("histogram", 0) > 0:
        score += 2

    if ema9 > ema21:
        score += 2
        reasons.append("EMA9 > EMA21 — short-term uptrend")
    else:
        score -= 2

    if bb.get("breakout_up"):
        score += 2
        reasons.append("Bollinger upper breakout — explosive momentum")

    if price > vwap:
        score += 1

    if rr >= 2.0:
        score += 2

    # Binary verdict for quick swing decisions
    if score >= 5:
        verdict     = "BUY"
        verdict_why = "Strong sudden spike + bullish confluence"
        confidence  = min(100, int((score / 12) * 100))
    else:
        verdict     = "SELL"
        verdict_why = "Weak spike or bearish signals — skip"
        confidence  = 0

    stop_loss = round(price - 1.5 * atr, 2) if atr else round(price * 0.95, 2)
    target_1  = round(price + 3.0 * atr, 2) if atr else round(price * 1.08, 2)
    target_2  = round(sr.get("nearest_resistance", price * 1.12), 2)
    risk_amt  = max(price - stop_loss, 0.01)
    rr_actual = round((target_1 - price) / risk_amt, 2)

    return {
        "verdict":        verdict,
        "verdict_reason": verdict_why,
        "score":          score,
        "confidence_pct": confidence,
        "entry_zone":     f"${price:.2f}–${round(price + atr * 0.5, 2):.2f}",
        "stop_loss":      stop_loss,
        "target_1":       target_1,
        "target_2":       target_2,
        "risk_reward":    rr_actual,
        "position_note":  _position_note(verdict, gap, rsi, volume_spike),
        "reasons":        reasons,
    }


def _position_note(verdict: str, gap: float, rsi: float, rvol: float) -> str:
    if verdict == "BUY":
        if gap > 12:
            return "Wait for first 5-min candle to close — large gap may fill"
        if rsi > 70:
            return "Enter on pullback to VWAP or EMA9"
        return "Enter at open or small dip • ATR stop below"
    else:
        return "Skip or short on breakdown • high risk of reversal"


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING (handles pre-market + after-hours automatically)
# ══════════════════════════════════════════════════════════════════════════════

def get_premarket_data(ticker: str) -> dict | None:
    try:
        tk = yf.Ticker(ticker)
        intra = tk.history(period="5d", interval="1m", prepost=True, auto_adjust=True)
        if intra.empty:
            return None
        intra.index = intra.index.tz_convert(ET)

        now_et = datetime.now(ET)
        today  = now_et.date()

        pm_start = datetime(today.year, today.month, today.day, 4,  0, tzinfo=ET)
        pm_end   = datetime(today.year, today.month, today.day, 9, 30, tzinfo=ET)
        premarket = intra[(intra.index >= pm_start) & (intra.index < pm_end)]

        session_label = "pre-market"
        if premarket.empty:
            prev = today - timedelta(days=1)
            ah_s = datetime(prev.year, prev.month, prev.day, 16, 0, tzinfo=ET)
            ah_e = datetime(prev.year, prev.month, prev.day, 20, 0, tzinfo=ET)
            premarket = intra[(intra.index >= ah_s) & (intra.index < ah_e)]
            session_label = "after-hours"

        if premarket.empty:
            return None

        session_volume   = int(premarket["Volume"].sum())
        session_open     = float(premarket["Open"].iloc[0])
        session_close    = float(premarket["Close"].iloc[-1])
        session_high     = float(premarket["High"].max())
        session_low      = float(premarket["Low"].min())
        price_change_pct = (session_close - session_open) / session_open * 100

        reg = intra[(intra.index.hour >= 9) & (intra.index.hour < 16)]
        avg_daily_vol = int(reg.groupby(reg.index.date)["Volume"].sum().mean()) if not reg.empty else session_volume

        volume_spike = round(session_volume / max(avg_daily_vol * 0.15, 1), 2)

        prev_reg = intra[(intra.index.hour >= 9) & (intra.index.hour < 16) & (intra.index.date < today)]
        prev_close = float(prev_reg["Close"].iloc[-1]) if not prev_reg.empty else session_open
        gap_pct = round((session_open - prev_close) / prev_close * 100, 2)

        daily = tk.history(period="60d", interval="1d", auto_adjust=True)
        if daily.empty:
            daily = pd.DataFrame({
                "Open":   [session_open],
                "High":   [session_high],
                "Low":    [session_low],
                "Close":  [session_close],
                "Volume": [float(session_volume)],
            })

        close = daily["Close"]
        high  = daily["High"]
        low   = daily["Low"]

        indicators = {
            "rsi":     calc_rsi(close),
            "macd":    calc_macd(close),
            "bollinger": calc_bollinger(close),
            "atr":     calc_atr(high, low, close),
            "ema9":    calc_ema(close, 9),
            "ema21":   calc_ema(close, 21),
            "vwap":    round(calc_vwap(reg) if not reg.empty else session_close, 2),
            "support_resistance": calc_support_resistance(daily),
        }

        data = {
            "ticker":                  ticker,
            "session":                 session_label,
            "session_open":            round(session_open, 2),
            "session_close":           round(session_close, 2),
            "session_high":            round(session_high, 2),
            "session_low":             round(session_low, 2),
            "price_change_pct":        round(price_change_pct, 2),
            "gap_from_prev_close_pct": gap_pct,
            "prev_close":              round(prev_close, 2),
            "session_volume":          session_volume,
            "avg_daily_volume":        avg_daily_vol,
            "volume_spike_x":          volume_spike,
            "daily_vol_profile":       [int(v) for v in reg.groupby(reg.index.date)["Volume"].sum().tail(5).tolist()],
            "indicators":              indicators,
            "scanned_at":              now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        }

        data["swing_trade"] = swing_score(data)
        return data

    except Exception as exc:
        print(f"  [WARN] {ticker}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SCANNING
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(watchlist: list[str], min_volume_spike: float = 5.0, min_price_change_pct: float = 7.0) -> tuple[list[dict], list[dict]]:
    all_results, hits = [], []

    for i, ticker in enumerate(watchlist, 1):
        print(f"  [{i:>2}/{len(watchlist)}] Scanning {ticker}…", end=" ", flush=True)
        data = get_premarket_data(ticker)

        if data is None:
            print("no data")
            continue

        all_results.append(data)

        vol_ok   = data["volume_spike_x"] >= min_volume_spike
        price_ok = abs(data["price_change_pct"]) >= min_price_change_pct

        if vol_ok and price_ok:
            hits.append(data)
            sw = data["swing_trade"]
            print(f"SPIKE  vol={data['volume_spike_x']}×  price={data['price_change_pct']:+.1f}%  → [{sw['verdict']}]")
        else:
            print(f"ok     vol={data['volume_spike_x']:.1f}×  price={data['price_change_pct']:+.1f}%")

    hits.sort(key=lambda x: x["swing_trade"]["score"], reverse=True)
    return all_results, hits


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

VERDICT_ICON = {"BUY": "🚀 BUY", "SELL": "❌ SELL"}


def print_report(hits: list[dict], all_results: list[dict], args) -> None:
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    W   = 85
    print("\n" + "═" * W)
    print(f"  SUDDEN SPIKE SWING SCANNER  —  {now}")
    print(f"  Thresholds: vol ≥ {args.min_vol}×  |  price move ≥ {args.min_price}%")
    print("═" * W)

    if not hits:
        print("  No sudden spikes met the thresholds right now.\n")
        return

    buy_ct  = sum(1 for h in hits if h["swing_trade"]["verdict"] == "BUY")
    sell_ct = len(hits) - buy_ct

    print(f"\n  {len(hits)} sudden spike(s)  |  BUY: {buy_ct}  SELL: {sell_ct}\n")

    for d in hits:
        sw   = d["swing_trade"]
        ind  = d["indicators"]
        icon = VERDICT_ICON[sw["verdict"]]
        arr  = "^" if d["price_change_pct"] >= 0 else "v"

        print("  "
