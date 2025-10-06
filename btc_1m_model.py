#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, math, sys, sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt

# -------- indicators --------
def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def rsi(x: pd.Series, period: int = 14) -> pd.Series:
    d = x.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill()

def macd(x: pd.Series, fast=12, slow=26, signal=9):
    ef, es = ema(x, fast), ema(x, slow)
    line = ef - es
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def zscore(x: pd.Series, window: int = 50):
    r = x.rolling(window)
    mu, sd = r.mean(), r.std(ddof=0)
    return (x - mu) / sd.replace(0, np.nan)

# -------- composite score --------
def compute_score(df: pd.DataFrame) -> float:
    if len(df) < 60: return -1e9
    close, vol = df["close"], df["volume"]

    e20, e50 = ema(close, 20), ema(close, 50)
    _, _, hist = macd(close, 12, 26, 9)
    rsi14, vz = rsi(close, 14), zscore(vol, 50)

    last = float(close.iloc[-1])
    ema_spread_pct = (float(e20.iloc[-1]) - float(e50.iloc[-1])) / last

    r = float(rsi14.iloc[-1])
    if 50 <= r <= 60:
        rsi_bonus = (r - 50) / 20.0
    elif r < 35:
        rsi_bonus = (35 - r) / 100.0
    else:
        rsi_bonus = 0.0

    vz_val = float(vz.iloc[-1]) if not math.isnan(float(vz.iloc[-1])) else 0.0
    vzb = max(0.0, min(vz_val, 3.0)) / 10.0

    atr_proxy = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    macd_contrib = 0.0 if pd.isna(atr_proxy) or atr_proxy == 0 else max(-1.0, min(1.0, float(hist.iloc[-1] / atr_proxy)))

    return float(2.0 * ema_spread_pct + 1.0 * macd_contrib + 0.5 * rsi_bonus + 1.0 * vzb)

def make_signal(score: float, hi: float, lo: float) -> str:
    return "BUY" if score >= hi else ("SELL" if score <= lo else "HOLD")

def backtest_onebar(df: pd.DataFrame, hi: float, lo: float, fee_bps: float = 5.0):
    if len(df) < 80:
        return {"acc": np.nan, "cumret_pct": np.nan, "n_buy":0, "n_sell":0, "n_hold":0, "last_signal":"HOLD", "last_score":np.nan}
    close = df["close"].astype(float)

    scores = []
    for i in range(len(df)):
        scores.append(np.nan if i < 60 else compute_score(df.iloc[:i+1]))
    scores = pd.Series(scores, index=df.index)

    sig = scores.apply(lambda s: make_signal(s, hi, lo))
    pos = sig.map({"BUY":1, "SELL":-1, "HOLD":0}).astype(float)

    ret = (close.shift(-1) - close) / close
    fee = fee_bps / 10000.0
    trade_ret = pos * ret - (np.abs(pos) * fee)
    trade_ret.iloc[-1] = 0.0
    pos.iloc[-1] = 0.0

    trade_mask = pos != 0
    acc = float((np.sign(pos[trade_mask]) == np.sign(ret[trade_mask])).mean()) if trade_mask.sum() > 0 else np.nan
    cumret = float((1.0 + trade_ret.fillna(0)).prod() - 1.0)

    return {
        "acc": acc,
        "cumret_pct": 100.0 * cumret,
        "n_buy": int((sig=="BUY").sum()),
        "n_sell": int((sig=="SELL").sum()),
        "n_hold": int((sig=="HOLD").sum()),
        "last_signal": sig.iloc[-1],
        "last_score": float(scores.iloc[-1]) if pd.notna(scores.iloc[-1]) else np.nan,
    }

# -------- data utils --------
def load_exchange(name: str):
    name = name.lower()
    if name in ("binance","binanceusdm"): return ccxt.binance({"enableRateLimit":True})
    if name == "okx": return ccxt.okx({"enableRateLimit":True})
    if name == "bybit": return ccxt.bybit({"enableRateLimit":True})
    if name == "kraken": return ccxt.kraken({"enableRateLimit":True})
    raise ValueError(f"Unsupported exchange: {name}")

def fetch_df(ex, symbol="BTC/USDT", timeframe="1m", limit=700):
    if not getattr(ex, "markets", None): ex.load_markets()
    time.sleep(getattr(ex, "rateLimit", 200)/1000)
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="BTC-only 1m model with BUY/SELL/HOLD and accuracy.")
    ap.add_argument("--exchange", default="binance", help="binance|okx|bybit|kraken")
    ap.add_argument("--symbol", default="BTC/USDT", help="BTC pair on chosen exchange")
    ap.add_argument("--limit", type=int, default=700, help="candles to fetch")
    ap.add_argument("--score_hi", type=float, default=0.01, help="BUY if score ≥ this")
    ap.add_argument("--score_lo", type=float, default=-0.01, help="SELL if score ≤ this")
    ap.add_argument("--fee_bps", type=float, default=5.0, help="round-trip fee bps")
    ap.add_argument("--db", default="crypto_signals.db", help="SQLite file to append")
    ap.add_argument("--csv", default="btc_1m_snapshot.csv", help="CSV snapshot filename")
    args = ap.parse_args()

    try:
        ex = load_exchange(args.exchange); ex.load_markets()
    except Exception as e:
        print(f"Exchange init failed: {e}"); sys.exit(1)

    if args.symbol not in ex.symbols:
        print(f"Symbol {args.symbol} not found on {ex.id}."); sys.exit(1)

    print(f"\nExchange: {ex.id} | Symbol: {args.symbol} | TF: 1m | Limit: {args.limit}")
    print(f"UTC Now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        df = fetch_df(ex, args.symbol, "1m", args.limit)
        if len(df) < 80: raise RuntimeError("Not enough candles.")
    except Exception as e:
        print(f"Data fetch error: {e}"); sys.exit(1)

    score = compute_score(df)
    signal = make_signal(score, args.score_hi, args.score_lo)
    bt = backtest_onebar(df, args.score_hi, args.score_lo, args.fee_bps)

    row = {
        "symbol": args.symbol,
        "price": float(df["close"].iloc[-1]),
        "score": float(score),
        "signal": signal,
        "acc": bt["acc"],
        "cumret%": bt["cumret_pct"],
        "BUYs": bt["n_buy"],
        "SELLs": bt["n_sell"],
        "HOLDs": bt["n_hold"],
    }

    # print table (single row)
    pd.set_option("display.float_format", lambda v: f"{v:,.6f}")
    cols = ["symbol","price","score","signal","acc","cumret%","BUYs","SELLs","HOLDs"]
    print("\nBTC 1m signal:")
    print(pd.DataFrame([row], columns=cols).to_string(index=False))

    # save to SQLite + CSV  (NOTE: use candle_limit instead of reserved 'limit')
    shown = pd.DataFrame([row], columns=cols)
    shown["run_ts_utc"] = pd.Timestamp.utcnow()
    shown["exchange"] = ex.id
    shown["timeframe"] = "1m"
    shown["candle_limit"] = args.limit
    shown["score_hi"] = args.score_hi
    shown["score_lo"] = args.score_lo
    shown = shown[["run_ts_utc","exchange","timeframe","candle_limit","score_hi","score_lo"] + cols]

    conn = sqlite3.connect(args.db)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS btc_signals (
      run_ts_utc  TEXT,
      exchange    TEXT,
      timeframe   TEXT,
      candle_limit INTEGER,
      score_hi    REAL,
      score_lo    REAL,
      symbol      TEXT,
      price       REAL,
      score       REAL,
      signal      TEXT,
      acc         REAL,
      "cumret%"   REAL,
      BUYs        INTEGER,
      SELLs       INTEGER,
      HOLDs       INTEGER
    );
    """)
    shown.to_sql("btc_signals", conn, if_exists="append", index=False)
    conn.close()
    print(f"\n✅ Saved to {args.db} (table: btc_signals)")
    shown.to_csv(args.csv, index=False)
    print(f"🗂️  CSV snapshot: {args.csv}")

if __name__ == "__main__":
    main()
