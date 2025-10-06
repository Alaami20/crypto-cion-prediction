#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import ccxt

# ---- indicators ----
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill()

def macd(s: pd.Series, fast=12, slow=26, signal=9):
    ef, es = ema(s, fast), ema(s, slow)
    line = ef - es
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

# ---- features & decision ----
def features(df: pd.DataFrame):
    close = df["close"].astype(float)
    e9, e21 = ema(close, 9), ema(close, 21)
    macd_line, macd_sig, macd_hist = macd(close, 12, 26, 9)
    rsi14 = rsi(close, 14)
    mom1 = close.pct_change(1)
    return {
        "price": float(close.iloc[-1]),
        "ema9_gt_21": 1.0 if e9.iloc[-1] > e21.iloc[-1] else 0.0,
        "ema_spread": float(e9.iloc[-1] - e21.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "rsi": float(rsi14.iloc[-1]),
        "mom1": float(mom1.iloc[-1]),
    }

def decide(feat, hi=0.60):
    """
    Majority vote (5 simple signals):
      up votes: ema9>21, ema_spread>0, macd_hist>0, rsi>50, mom1>0
    prob_up = votes/5
    """
    votes = [
        1 if feat["ema9_gt_21"] > 0.5 else 0,
        1 if feat["ema_spread"]  > 0   else 0,
        1 if feat["macd_hist"]   > 0   else 0,
        1 if feat["rsi"]         > 50  else 0,
        1 if feat["mom1"]        > 0   else 0,
    ]
    prob_up = sum(votes) / 5.0

    if prob_up >= hi:
        return "BUY", prob_up   # expect UP next minute (CALL)
    elif prob_up <= (1.0 - hi):
        return "SELL", 1.0 - prob_up  # expect DOWN next minute (PUT)
    else:
        return "SKIP", 1.0 - abs(prob_up - 0.5) * 2  # low confidence

# ---- data ----
def load_exchange(name="binance"):
    name = name.lower()
    if name in ("binance","binanceusdm"): return ccxt.binance({"enableRateLimit": True})
    if name == "okx": return ccxt.okx({"enableRateLimit": True})
    if name == "bybit": return ccxt.bybit({"enableRateLimit": True})
    if name == "kraken": return ccxt.kraken({"enableRateLimit": True})
    raise ValueError("Unsupported exchange")

def fetch_df(ex, symbol="BTC/USDT", limit=200):
    if not getattr(ex, "markets", None): ex.load_markets()
    time.sleep(getattr(ex, "rateLimit", 200)/1000)
    o = ex.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="BTC 1m action: BUY (up next minute) or SELL (down next minute) or SKIP.")
    ap.add_argument("--exchange", default="binance", help="binance|okx|bybit|kraken")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--limit", type=int, default=200, help="candles history")
    ap.add_argument("--hi", type=float, default=0.60, help="confidence threshold (0.60 = 3/5 votes)")
    args = ap.parse_args()

    ex = load_exchange(args.exchange); ex.load_markets()
    if args.symbol not in ex.symbols:
        print(f"Symbol {args.symbol} not found on {ex.id}."); sys.exit(1)

    df = fetch_df(ex, args.symbol, args.limit)
    if len(df) < 70:
        print("Not enough candles."); sys.exit(1)

    feat = features(df)
    action, conf = decide(feat, hi=args.hi)

    # timing: enter now, exit next minute close
    now_utc = datetime.now(timezone.utc)
    # round up to next minute boundary for exit
    exit_utc = (now_utc.replace(second=0, microsecond=0) + timedelta(minutes=1))

    price = feat["price"]
    direction = "UP next 1m" if action == "BUY" else ("DOWN next 1m" if action == "SELL" else "NO TRADE")

    # clear, explicit output
    print("\n=== BTC 1m Decision ===")
    print(f"Time (UTC):     {now_utc:%Y-%m-%d %H:%M:%S}")
    print(f"Symbol:         {args.symbol}  on {ex.id}")
    print(f"Current price:  {price:,.2f}")
    print(f"ACTION:         {action}   ← {direction}")
    print(f"Confidence:     {conf:.2f} (threshold {args.hi:.2f})")
    print(f"Entry:          NOW (at current market)")
    print(f"Exit:           At next 1m candle close  ({exit_utc:%Y-%m-%d %H:%M:%S} UTC)")

    # quick explanation (why this action)
    print("\nReasons (votes out of 5):")
    votes = {
        "EMA9>EMA21": int(feat['ema9_gt_21'] > 0.5),
        "EMA spread>0": int(feat['ema_spread'] > 0),
        "MACD hist>0": int(feat['macd_hist'] > 0),
        "RSI>50": int(feat['rsi'] > 50),
        "Mom(1)>0": int(feat['mom1'] > 0),
    }
    print(" " + ", ".join([f"{k}={v}" for k,v in votes.items()]))

if __name__ == "__main__":
    main()
