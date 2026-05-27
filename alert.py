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
    'VWRA.L',   # Vanguard FTSE All-World UCITS ETF
    'MSFT',     # Microsoft
    'NVDA',     # Nvidia
    'GOOGL',    # Alphabet
    'AMZN',     # Amazon
    'AMD',      # AMD
    'NVO',      # Novo Nordisk
    'META',     # Meta
    'BABA',     # Alibaba
    'RKLB',     # Rocket Lab
    'MU',       # Micron
    'GEV',      # GE Vernova
    'VRT',        # Vertiv Holding
    'VST'
]

RSI_PERIOD      = 14
RSI_THRESHOLD   = 40      # Alert when RSI drops below this
EMA_PERIOD      = 200     # EMA period to watch

# Tickers to exclude from EMA200 signals (RSI alerts still apply)
# Reason: CSPX is a DCA target — we buy on RSI dips only, not EMA bounces
EMA_EXCLUDE = {
    'CSPX.L',   # DCA target — RSI dip only
    'VWRA.L',   # DCA target — RSI dip only
}
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
        # Use 5y for LSE-listed ETFs (VWRA.L, CSPX.L) — yfinance sometimes
        # returns sparse data for London exchange tickers on shorter periods,
        # leaving too few rows to calculate RSI(14) and EMA(200) without NaNs.
        period = "5y" if ticker.endswith(".L") else "2y"
        df = yf.download(ticker, period=period, interval="1d", progress=False)
    except Exception as e:
        print(f"[{now()}] Error downloading {ticker}: {e}")
        return None

    if df.empty:
        print(f"[{now()}] No data returned for {ticker}, skipping.")
        return None

    close = df["Close"].squeeze()

    # Drop any NaN prices (can happen with LSE tickers on non-trading days)
    close = close.dropna()

    print(f"[{now()}] {ticker}: {len(close)} rows of price data available.")

    if len(close) < EMA_PERIOD + RSI_PERIOD:
        print(f"[{now()}] {ticker}: Not enough rows ({len(close)}) "
              f"— need at least {EMA_PERIOD + RSI_PERIOD}, skipping.")
        return None

    rsi    = calc_rsi(close)
    ema200 = calc_ema(close)

    # Drop NaN values produced during indicator warmup period
    rsi    = rsi.dropna()
    ema200 = ema200.dropna()

    if rsi.empty or ema200.empty:
        print(f"[{now()}] {ticker}: RSI or EMA200 still NaN after calculation, skipping.")
        return None

    # Align all series to the same index after dropna
    common_idx = close.index.intersection(rsi.index).intersection(ema200.index)
    close  = close.loc[common_idx]
    rsi    = rsi.loc[common_idx]
    ema200 = ema200.loc[common_idx]

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

    # EMA200 signals — suppressed for DCA tickers (ETFs bought on RSI dips only)
    ema_excluded = ticker in EMA_EXCLUDE

    # EMA200 bounce: price was below EMA, now crossed above (buy signal)
    ema_bounce = (
        not ema_excluded and
        (prev_price < prev_ema) and (latest_price >= latest_ema)
    )

    # Price proximity to EMA200 (within 1% — approaching bounce zone)
    ema_proximity = (
        not ema_excluded and
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

# Plain English explanations for each signal — what it means and what to do
SIGNAL_EXPLANATIONS = {
    "rsi_alert": {
        "icon"   : "⚠️",
        "title"  : "RSI JUST CROSSED BELOW 40",
        "meaning": "The stock has just entered oversold territory. "
                   "Selling pressure has been heavy and the stock may be due for a bounce. "
                   "This is not a guarantee — but historically, RSI below 40 signals the "
                   "market has overreacted and a recovery is more likely than not.",
        "action" : "Watch closely. Consider a small initial add if your conviction is high. "
                   "Wait for a second confirmation (e.g. EMA200 bounce) before going bigger.",
    },
    "rsi_low": {
        "icon"   : "🟠",
        "title"  : "RSI BELOW 40 — STILL OVERSOLD",
        "meaning": "The stock has been in oversold territory for more than one day. "
                   "Persistent low RSI means sellers are still in control, but the longer "
                   "it stays this low, the more compressed the spring — a sharp recovery "
                   "is building up pressure.",
        "action" : "Hold your position if you own it. If you've been waiting to add, "
                   "this is an accumulation zone. Don't go all-in — buy in tranches.",
    },
    "ema_bounce": {
        "icon"   : "📈",
        "title"  : "EMA200 BOUNCE — MAJOR SUPPORT RECLAIMED",
        "meaning": "The stock was trading below its 200-day moving average (a key long-term "
                   "support level) and has just crossed back above it. This is one of the "
                   "most widely watched signals by institutional investors. It means the "
                   "long-term downtrend may be reversing. RSI does not need to be below 40 "
                   "for this to be meaningful — the price action alone is the signal.",
        "action" : "This is a buy signal on its own. Consider adding to your position. "
                   "For MSFT specifically: this is the confirmation you have been waiting for. "
                   "A partial add here is justified even if RSI is neutral (40-60).",
    },
    "ema_proximity": {
        "icon"   : "👀",
        "title"  : "APPROACHING EMA200 — KEY LEVEL NEARBY",
        "meaning": "The stock is trading within 1% above its 200-day moving average. "
                   "This level acts as a floor for many institutional buy orders. "
                   "The stock is at a decision point — it either bounces here (bullish) "
                   "or breaks below (bearish, more downside likely).",
        "action" : "Do not buy yet. Wait and watch. If it holds above EMA200 tomorrow, "
                   "that confirms support and you can add. If it breaks below, stand aside.",
    },
}


def signal_strength(result: dict) -> tuple[int, str]:
    """
    Count active signals and return a strength score + plain English summary.
    The more signals firing together, the higher the conviction.
    """
    active = []
    if result["rsi_alert"] or result["rsi_low"]:
        active.append("RSI oversold")
    if result["ema_bounce"]:
        active.append("EMA200 reclaimed")
    if result["ema_proximity"]:
        active.append("EMA200 proximity")

    count = len(active)

    if count >= 2:
        label = (
            "🔥 <b>STRONG SIGNAL — Multiple confirmations align</b>\n"
            "<i>RSI is oversold AND the stock is at/reclaiming EMA200. "
            "This combination historically offers the best risk/reward entry. "
            "High conviction add — consider sizing in meaningfully.</i>"
        )
    elif count == 1:
        label = (
            "💡 <b>SINGLE SIGNAL — One confirmation, monitor closely</b>\n"
            "<i>Only one indicator is firing. Valid reason to watch and prepare capital, "
            "but wait for a second confirmation before making a large move. "
            "A small starter position is reasonable.</i>"
        )
    else:
        label = ""

    return count, label


def format_alert_block(result: dict) -> str:
    """Format a single ticker's alert as a Telegram HTML block with plain English."""
    ticker = result["ticker"]
    price  = result["price"]
    rsi    = result["rsi"]
    ema    = result["ema200"]

    # RSI visual indicator
    if rsi < 30:
        rsi_bar = "🔴🔴🔴 Extremely oversold"
    elif rsi < 40:
        rsi_bar = "🟠🟠⚪ Oversold"
    elif rsi < 60:
        rsi_bar = "🟡🟡⚪ Neutral"
    else:
        rsi_bar = "🟢🟢🟢 Strong momentum"

    lines = [
        f"<b>{'─' * 22}</b>",
        f"📌 <b>{ticker}</b>",
        f"💰 Price   : <b>${price}</b>",
        f"📊 RSI(14) : <b>{rsi}</b> — {rsi_bar}",
        f"📉 EMA200  : <b>${ema}</b>" +
        (" <i>(EMA signals off — DCA ticker)</i>" if ticker in EMA_EXCLUDE else ""),
    ]

    # Signal strength header
    _, strength_label = signal_strength(result)
    if strength_label:
        lines.append(f"\n{strength_label}")

    # Individual signal blocks with plain English
    lines.append("")  # spacer

    if result["rsi_alert"]:
        s = SIGNAL_EXPLANATIONS["rsi_alert"]
        lines.append(f"{s['icon']} <b>{s['title']}</b>")
        lines.append(f"<i>What this means:</i> {s['meaning']}")
        lines.append(f"<i>What to do:</i> {s['action']}")

    elif result["rsi_low"]:
        s = SIGNAL_EXPLANATIONS["rsi_low"]
        lines.append(f"{s['icon']} <b>{s['title']}</b>")
        lines.append(f"<i>What this means:</i> {s['meaning']}")
        lines.append(f"<i>What to do:</i> {s['action']}")

    if result["ema_bounce"]:
        s = SIGNAL_EXPLANATIONS["ema_bounce"]
        lines.append(f"{s['icon']} <b>{s['title']}</b>")
        lines.append(f"<i>What this means:</i> {s['meaning']}")
        lines.append(f"<i>What to do:</i> {s['action']}")

    if result["ema_proximity"]:
        s = SIGNAL_EXPLANATIONS["ema_proximity"]
        lines.append(f"{s['icon']} <b>{s['title']}</b>")
        lines.append(f"<i>What this means:</i> {s['meaning']}")
        lines.append(f"<i>What to do:</i> {s['action']}")

    return "\n".join(lines)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S SGT")
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
            result["rsi_alert"]     or
            result["rsi_low"]       or
            result["ema_bounce"]    or   # EMA200 bounce alone is a valid signal
            result["ema_proximity"]      # Approaching EMA200 is worth knowing
        )
        is_watching = False  # All signals now promote directly to triggered

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
        f"🗓  {datetime.now(ZoneInfo('America/New_York')).strftime('%A, %d %b %Y')}",
        f"⏰  {datetime.now(ZoneInfo('America/New_York')).strftime('%H:%M')} UTC-4 (NY)"
        f"  |  {datetime.now(ZoneInfo('Asia/Singapore')).strftime('%H:%M')} SGT",
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
        "\n<i>"
        "RSI &lt; 40 = stock is oversold, sellers have overreacted | "
        "EMA200 bounce = price reclaimed key long-term support | "
        "Both together = strongest possible buy signal"
        "</i>"
    )

    full_message = "\n".join(message_parts)
    send_telegram(full_message)
    print(f"[{now()}] Done.")


if __name__ == "__main__":
    run()
# ─────────────────────────────────────────────────────────────────────────────
