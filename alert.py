import os
import json
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
STOCKS = [
    'CSPX.L',   # iShares Core S&P 500 UCITS ETF
    'MSFT',     # Microsoft
    'NVDA',     # Nvidia
    'GOOGL',    # Alphabet
    'AMZN',     # Amazon
]

RSI_PERIOD      = 14
RSI_THRESHOLD   = 40      # Alert when RSI drops below this
EMA_PERIOD      = 200     # EMA period to watch
# ─────────────────────────────────────────────────────────────────────────────


# ── TELEGRAM ─────────────────────────────────────────────────────────────────
def send_telegram(message: str) -> None:
    """Send a message via Telegram bot."""
    token   = os.environ['TELEGRAM_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id"    : chat_id,
        "text"       : message,
        "parse_mode" : "HTML"
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data    = payload,
        headers = {"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"[{now()}] Telegram message sent.")
    except Exception as e:
        print(f"[{now()}] Failed to send Telegram message: {e}")
# ─────────────────────────────────────────────────────────────────────────────


# ── INDICATORS ───────────────────────────────────────────────────────────────
def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def calc_ema(close: pd.Series, period: int = EMA_PERIOD) -> pd.Series:
    """Calculate EMA."""
    return close.ewm(span=period, adjust=False).mean()
# ─────────────────────────────────────────────────────────────────────────────


# ── SIGNAL CHECK ─────────────────────────────────────────────────────────────
def check_signals(ticker: str) -> dict | None:
    """
    Download price data and check for:
      1. RSI(14) dropping below RSI_THRESHOLD
      2. Price bouncing off EMA(200) — crossed above after being below
    Returns a result dict, or None if data is unavailable.
    """
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
    except Exception as e:
        print(f"[{now()}] Error downloading {ticker}: {e}")
        return None

    if df.empty or len(df) < EMA_PERIOD + RSI_PERIOD:
        print(f"[{now()}] Not enough data for {ticker}, skipping.")
        return None

    close = df["Close"].squeeze()

    rsi    = calc_rsi(close)
    ema200 = calc_ema(close)

    # Latest and previous values
    latest_price = float(close.iloc[-1])
    prev_price   = float(close.iloc[-2])
    latest_ema   = float(ema200.iloc[-1])
    prev_ema     = float(ema200.iloc[-2])
    latest_rsi   = float(rsi.iloc[-1])
    prev_rsi     = float(rsi.iloc[-2])

    # Signal conditions
    # RSI crossed below threshold (was above, now below)
    rsi_alert = (prev_rsi >= RSI_THRESHOLD) and (latest_rsi < RSI_THRESHOLD)

    # RSI already below threshold (persistent low RSI — still worth knowing)
    rsi_low = latest_rsi < RSI_THRESHOLD

    # EMA200 bounce: price was below EMA, now crossed above (buy signal)
    ema_bounce = (prev_price < prev_ema) and (latest_price >= latest_ema)

    # Price proximity to EMA200 (within 1% — approaching bounce zone)
    ema_proximity = (
        latest_price >= latest_ema and
        latest_price <= latest_ema * 1.01
    )

    return {
        "ticker"        : ticker,
        "price"         : round(latest_price, 2),
        "rsi"           : round(latest_rsi, 2),
        "ema200"        : round(latest_ema, 2),
        "rsi_alert"     : rsi_alert,       # RSI just crossed below 40
        "rsi_low"       : rsi_low,         # RSI currently below 40
        "ema_bounce"    : ema_bounce,       # Price just crossed above EMA200
        "ema_proximity" : ema_proximity,    # Price hugging EMA200 from above
    }
# ─────────────────────────────────────────────────────────────────────────────


# ── FORMATTING ────────────────────────────────────────────────────────────────
def format_alert_block(result: dict) -> str:
    """Format a single ticker's alert as a Telegram HTML block."""
    ticker = result["ticker"]
    price  = result["price"]
    rsi    = result["rsi"]
    ema    = result["ema200"]

    # RSI emoji indicator
    if rsi < 30:
        rsi_bar = "🔴🔴🔴"
    elif rsi < 40:
        rsi_bar = "🟠🟠⚪"
    else:
        rsi_bar = "🟢🟢🟢"

    lines = [f"<b>{'─'*20}</b>",
             f"📌 <b>{ticker}</b>",
             f"💰 Price  : <b>${price}</b>",
             f"📊 RSI(14): <b>{rsi}</b>  {rsi_bar}",
             f"📉 EMA200 : <b>${ema}</b>"]

    if result["rsi_alert"]:
        lines.append("⚠️  <b>RSI CROSSED BELOW 40 — Potential buy zone</b>")
    elif result["rsi_low"]:
        lines.append("⚠️  <b>RSI BELOW 40 — Oversold territory</b>")

    if result["ema_bounce"]:
        lines.append("📈 <b>EMA200 BOUNCE — Price reclaimed 200-day EMA</b>")

    if result["ema_proximity"]:
        lines.append("👀 <b>Price hugging EMA200 — Watch closely</b>")

    return "\n".join(lines)


def now() -> str:
    return datetime.now(ZoneInfo("Singapore")).strftime("%Y-%m-%d %H:%M:%S")
# ─────────────────────────────────────────────────────────────────────────────


# ── MAIN ─────────────────────────────────────────────────────────────────────
def run() -> None:
    print(f"[{now()}] Running stock signal check for {len(STOCKS)} tickers...")

    triggered    = []   # Stocks with active signals
    watchlist    = []   # Stocks near EMA200 but no full signal yet

    for ticker in STOCKS:
        result = check_signals(ticker)
        if result is None:
            continue

        has_signal = (
            result["rsi_alert"] or
            result["rsi_low"]   or
            result["ema_bounce"]
        )
        is_watching = result["ema_proximity"]

        if has_signal:
            triggered.append(result)
            print(f"  ✅ {ticker}: Signal triggered (RSI={result['rsi']}, "
                  f"EMA bounce={result['ema_bounce']})")
        elif is_watching:
            watchlist.append(result)
            print(f"  👀 {ticker}: Near EMA200 — watching")
        else:
            print(f"  ➖ {ticker}: No signal (RSI={result['rsi']})")

    # ── BUILD MESSAGE ─────────────────────────────────────────────────
    if not triggered and not watchlist:
        print(f"[{now()}] No signals today. No message sent.")
        return

    message_parts = [
        f"🔔 <b>Stock Signal Report</b>",
        f"🗓  {datetime.now(ZoneInfo("America/New_York")).strftime('%A, %d %b %Y')}",
        f"⏰  {datetime.now(ZoneInfo("America/New_York")).strftime('%H:%M')} UTC-4",
    ]

    if triggered:
        message_parts.append(f"\n🚨 <b>ACTION SIGNALS ({len(triggered)})</b>")
        for r in triggered:
            message_parts.append(format_alert_block(r))

    if watchlist:
        message_parts.append(f"\n👁  <b>WATCHLIST — Near EMA200 ({len(watchlist)})</b>")
        for r in watchlist:
            message_parts.append(
                f"• <b>{r['ticker']}</b> — "
                f"Price: ${r['price']} | "
                f"RSI: {r['rsi']} | "
                f"EMA200: ${r['ema200']}"
            )

    message_parts.append(
        "\n<i>Strategy: RSI &lt; 40 = oversold zone | "
        "EMA200 bounce = trend reclaim signal</i>"
    )

    full_message = "\n".join(message_parts)
    send_telegram(full_message)
    print(f"[{now()}] Done.")


if __name__ == "__main__":
    run()
# ─────────────────────────────────────────────────────────────────────────────
