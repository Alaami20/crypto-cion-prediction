#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt
import sys

# ========= Indicators =========
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def zscore(series: pd.Series, window: int = 50):
    roll = series.rolling(window)
    mean = roll.mean()
    std = roll.std(ddof=0)
    return (series - mean) / (std.replace(0, np.nan))

# ========= Composite score =========
def compute_score(df: pd.DataFrame) -> float:
    close = df["close"]
    vol = df["volume"]
    if len(df) < 60:
        return -1e9

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    _, _, hist = macd(close, 12, 26, 9)
    rsi14 = rsi(close, 14)
    vz = zscore(vol, 50)

    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    last_close = float(close.iloc[-1])
    hist_last = float(hist.iloc[-1])
    rsi_last = float(rsi14.iloc[-1])
    vz_last = float(vz.iloc[-1]) if not math.isnan(float(vz.iloc[-1])) else 0.0

    ema_spread_pct = (e20 - e50) / last_close

    rsi_bonus = 0.0
    if 50 <= rsi_last <= 60:
        rsi_bonus = (rsi_last - 50) / 20.0
    elif rsi_last < 35:
        rsi_bonus = (35 - rsi_last) / 100.0

    vz_bonus = max(0.0, min(vz_last, 3.0)) / 10.0

    atr_proxy = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    macd_contrib = 0.0
    if pd.notna(atr_proxy) and atr_proxy != 0:
        macd_contrib = max(-1.0, min(1.0, (hist_last / atr_proxy)))

    score = 2.0 * ema_spread_pct + 1.0 * macd_contrib + 0.5 * rsi_bonus + 1.0 * vz_bonus
    return float(score)

# ========= Signal & backtest =========
def make_signal(score: float, hi: float, lo: float) -> str:
    if score >= hi:
        return "BUY"
    elif score <= lo:
        return "SELL"
    else:
        return "HOLD"

def backtest_onebar(df: pd.DataFrame, hi: float, lo: float, fee_bps: float = 5.0):
    """
    One-bar horizon backtest:
    - signal from composite score on each bar (BUY/SELL/HOLD)
    - enter at bar close, exit next bar close
    - fee_bps is applied per round turn (approx: entry+exit)
    Returns: dict with acc, cumret%, counts.
    """
    close = df["close"].astype(float).copy()
    if len(close) < 80:
        return {"acc": np.nan, "cumret_pct": np.nan, "n_buy": 0, "n_sell": 0, "n_hold": 0}

    # Build rolling score to avoid lookahead
    scores = []
    for i in range(len(df)):
        if i < 60:
            scores.append(np.nan)
            continue
        sub = df.iloc[: i + 1]
        scores.append(compute_score(sub))
    scores = pd.Series(scores, index=df.index)

    sig = scores.apply(lambda s: make_signal(s, hi, lo))
    # Position for next bar:
    pos = sig.map({"BUY": 1, "SELL": -1, "HOLD": 0}).astype(float)

    # Next-bar returns
    next_close = close.shift(-1)
    ret = (next_close - close) / close  # simple return
    # PnL per bar on trades; subtract round-trip fee (bps -> fraction)
    fee = fee_bps / 10000.0
    trade_ret = pos * ret - (np.abs(pos) * fee)  # charge only when pos != 0

    # We cannot realize last bar’s next_close
    trade_ret.iloc[-1] = 0.0
    pos.iloc[-1] = 0.0

    # Accuracy on trades (ignore HOLD rows)
    trade_mask = pos != 0
    if trade_mask.sum() > 0:
        direction_ok = np.sign(pos[trade_mask]) == np.sign(ret[trade_mask])
        acc = float(direction_ok.mean())
    else:
        acc = np.nan

    cumret = float((1.0 + trade_ret.fillna(0)).prod() - 1.0)

    return {
        "acc": acc,
        "cumret_pct": 100.0 * cumret,
        "n_buy": int((sig == "BUY").sum()),
        "n_sell": int((sig == "SELL").sum()),
        "n_hold": int((sig == "HOLD").sum()),
        "last_signal": sig.iloc[-1],
        "last_score": float(scores.iloc[-1]) if pd.notna(scores.iloc[-1]) else np.nan,
    }

# ========= CCXT utils =========
def load_exchange(name: str):
    name = name.lower()
    if name in ("binance", "binanceusdm"):
        return ccxt.binance({"enableRateLimit": True})
    if name in ("okx",):
        return ccxt.okx({"enableRateLimit": True})
    if name in ("bybit",):
        return ccxt.bybit({"enableRateLimit": True})
    if name in ("kraken",):
        return ccxt.kraken({"enableRateLimit": True})
    raise ValueError(f"Unsupported exchange: {name}")

def safe_fetch_ohlcv(ex, symbol, timeframe="1m", limit=600):
    if not getattr(ex, "markets", None):
        ex.load_markets()
    time.sleep(ex.rateLimit / 1000 if hasattr(ex, "rateLimit") else 0.2)
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# ========= Main =========
def main():
    parser = argparse.ArgumentParser(description="Top 5 minute crypto signals with BUY/SELL/HOLD and accuracy.")
    parser.add_argument("--exchange", default="binance", help="binance | okx | bybit | kraken")
    parser.add_argument("--limit", type=int, default=700, help="1m candles to fetch (default 700)")
    parser.add_argument("--topk", type=int, default=5, help="how many rows to show")
    parser.add_argument("--score_hi", type=float, default=0.01, help="BUY threshold (default 0.01)")
    parser.add_argument("--score_lo", type=float, default=-0.01, help="SELL threshold (default -0.01)")
    parser.add_argument("--fee_bps", type=float, default=5.0, help="round-trip fee in bps (default 5 = 0.05%)")
    parser.add_argument("--symbols", nargs="*", default=[
        "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
        "TON/USDT","ADA/USDT","DOGE/USDT","LINK/USDT","AVAX/USDT",
        "TRX/USDT","NEAR/USDT","PEPE/USDT","MATIC/USDT","SUI/USDT"
    ])
    args = parser.parse_args()

    try:
        ex = load_exchange(args.exchange)
        ex.load_markets()
    except Exception as e:
        print(f"Exchange init failed: {e}")
        sys.exit(1)

    supported = [s for s in args.symbols if s in ex.symbols]
    if not supported:
        print("No provided symbols found on this exchange.")
        sys.exit(1)

    print(f"\nExchange: {ex.id} | TF: 1m | Limit: {args.limit} | Fee: {args.fee_bps} bps")
    print(f"UTC Now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    rows = []

    for sym in supported:
        try:
            ohlcv = safe_fetch_ohlcv(ex, sym, timeframe="1m", limit=args.limit)
            if not ohlcv or len(ohlcv) < 80:
                continue
            df = to_df(ohlcv)

            # Latest composite score & discrete signal on full df
            last_score = compute_score(df)
            last_signal = make_signal(last_score, args.score_hi, args.score_lo)

            # Backtest over whole window (no lookahead)
            bt = backtest_onebar(df, args.score_hi, args.score_lo, args.fee_bps)

            rows.append({
                "symbol": sym,
                "price": float(df["close"].iloc[-1]),
                "score": float(last_score),
                "signal": last_signal,
                "acc": bt["acc"],
                "cumret%": bt["cumret_pct"],
                "BUYs": bt["n_buy"],
                "SELLs": bt["n_sell"],
                "HOLDs": bt["n_hold"],
            })
        except ccxt.BaseError as ce:
            print(f"[{sym}] ccxt error: {ce}")
        except Exception as e:
            print(f"[{sym}] error: {e}")

    if not rows:
        print("No data collected.")
        sys.exit(0)

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan)

    # Rank by current score (strongest first)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    # Pretty print
    pd.set_option("display.float_format", lambda v: f"{v:,.6f}")
    display_cols = ["symbol","price","score","signal","acc","cumret%","BUYs","SELLs","HOLDs"]
    topk = max(1, min(args.topk, len(out)))

    print("\nTop signals (1m):")
    print(out.loc[:topk-1, display_cols].to_string(index=False))

    print("\nNotes:")
    print("• signal uses thresholds: BUY if score ≥ score_hi, SELL if score ≤ score_lo, else HOLD.")
    print("• acc = directional win-rate on BUY/SELL trades for a 1-bar horizon.")
    print("• cumret% = cumulative return of the 1-bar strategy (after round-trip fee).")
    print("• Heuristic only. Not financial advice.")

if __name__ == "__main__":
    main()
