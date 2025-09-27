# top5_hour_crypto_signals.py
# 1-hour multi-coin signals (BTC, ETH, BNB, SOL, XRP) with "big-move + confidence" strategy.
# Prints Buy/Sell/Hold for the latest closed hour, + per-coin metrics.
# Educational use only. Markets are risky.

import time, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import math
import matplotlib.pyplot as plt

try:
    import ccxt
except ImportError as e:
    raise SystemExit("ccxt is required. Install with: pip install ccxt") from e

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

# ========= knobs you can tweak =========
EXCHANGE  = "binance"             # public OHLCV (no API key needed)
SYMBOLS   = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT"]  # top 5 by liquidity
TIMEFRAME = "1h"
LOOKBACK_HOURS = 2000             # ~83 days of 1h candles
H        = 1                      # predict next 1h
K        = 0.0035                 # "big move" threshold (0.35% over 1h) -> tune 0.3%..0.7%
HI       = 0.70                   # Buy if P(up) >= HI
LO       = 0.30                   # Sell if P(up) <= LO
FEE      = 0.0010                 # 10 bps per side (adjust to your venue)
SEED     = 42

# ========= indicators (lightweight) =========
def rsi(s, w=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/w, adjust=False).mean()
    rd = dn.ewm(alpha=1/w, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100/(1+rs))

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def macd(s):
    m = ema(s,12) - ema(s,26)
    return m, ema(m,9)
def sma(s, w): return s.rolling(w).mean()

# ========= data fetch (ccxt, with pagination) =========
def fetch_ohlcv_df(ex, symbol, timeframe=TIMEFRAME, lookback_hours=LOOKBACK_HOURS, limit_per_call=1500):
    ms_per_bar = ex.parse_timeframe(timeframe) * 1000
    since = int((datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).timestamp() * 1000)
    out = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
        if not batch:
            break
        out.extend(batch)
        # advance 'since' to last+1 bar to avoid duplicates
        since = batch[-1][0] + ms_per_bar
        # stop if we reached near-now
        if len(batch) < limit_per_call:
            break
        # gentle rate limit
        time.sleep(ex.rateLimit / 1000.0)
        # also stop if we've clearly exceeded the window
        if since > int(time.time() * 1000):
            break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["ts","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[["Date","Open","High","Low","Close","Volume"]].dropna()
    # keep only fully closed candles (drop last if it's still forming)
    now_ms = int(time.time()*1000)
    bar_ms = ex.parse_timeframe(timeframe) * 1000
    if len(df) and (df["Date"].iloc[-1].value // 10**6 + bar_ms > now_ms):
        df = df.iloc[:-1]
    return df.reset_index(drop=True)

# ========= features / labels =========
def build_features(df: pd.DataFrame):
    x = df.copy()
    x["ret_1"] = x["Close"].pct_change()
    x["ret_3"] = x["Close"].pct_change(3)
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_48"] = x["ret_1"].rolling(48).std()
    x["rsi_14"] = rsi(x["Close"], 14)
    x["macd"], x["macd_sig"] = macd(x["Close"])
    x["sma_20"] = sma(x["Close"], 20)
    x["sma_50"] = sma(x["Close"], 50)
    x["sma_ratio"] = x["sma_20"] / (x["sma_50"] + 1e-12)
    for k in [1,2,3,6]:
        x[f"ret_lag{k}"] = x["ret_1"].shift(k)
    # future return over H bars (here H=1)
    x["fut_ret_H"] = x["Close"].shift(-H) / x["Close"] - 1.0
    # big-move mask
    mask = x["fut_ret_H"].abs() >= K
    big = x.loc[mask].copy()
    big["y"] = (big["fut_ret_H"] > 0).astype(int)
    feat_cols = ["ret_1","ret_3","vol_10","vol_48","rsi_14","macd","macd_sig",
                 "sma_ratio","ret_lag1","ret_lag2","ret_lag3","ret_lag6"]
    X = big[feat_cols].select_dtypes(include=[np.number]).copy()
    # mild downcast to float32 for speed/memory without overflow
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    y = big["y"].astype(int)
    return big, X, y

# ========= training / evaluation per coin =========
def fit_and_signal(df, symbol):
    if len(df) < 400:
        return {"symbol": symbol, "error": "not enough data"}
    data, X, y = build_features(df)
    if len(X) < 200:
        return {"symbol": symbol, "error": "not enough big-move samples"}

    split = int(len(X) * 0.8)  # time-based split
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]
    dates_te   = data["Date"].iloc[split:].reset_index(drop=True)
    close_te   = data["Close"].iloc[split:].reset_index(drop=True)
    futret_te  = data["fut_ret_H"].iloc[split:].reset_index(drop=True)

    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.07,
                                           max_iter=500, l2_regularization=1.0,
                                           random_state=SEED)
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:,1]
    pred_te  = (proba_te >= 0.5).astype(int)
    acc_big  = accuracy_score(y_te, pred_te)  # accuracy on big-move hours only

    # Confident signals only
    sig = np.zeros_like(proba_te, dtype=int)   # 1=BUY, -1=SELL, 0=HOLD
    sig[proba_te >= HI] = 1
    sig[proba_te <= LO] = -1
    take = sig != 0

    # Accuracy & precision on trades only
    if take.sum() > 0:
        pred_tr = np.where(sig[take]==1, 1, 0)
        y_tru   = y_te[take].to_numpy()
        acc_tr  = accuracy_score(y_tru, pred_tr)
        prec_up = precision_score(y_tru, pred_tr, pos_label=1)
        prec_dn = precision_score(1 - y_tru, 1 - pred_tr, pos_label=1)
    else:
        acc_tr = prec_up = prec_dn = np.nan

    # Simple trade PnL (enter at close, exit after H)
    rets = np.zeros_like(proba_te, dtype=float)
    rets[sig==1]  = futret_te[sig==1].to_numpy()
    rets[sig==-1] = -futret_te[sig==-1].to_numpy()
    rets_adj = rets - 2*FEE*(sig!=0).astype(float)

    def _sharpe(x, periods=24*365//6):  # rough scaling for hourly; ~1460/year, but ok
        x = pd.Series(x).replace([np.inf,-np.inf], np.nan).dropna()
        return np.nan if x.std()==0 else np.sqrt(periods)*x.mean()/x.std()

    mean_trade = float(np.nan if take.sum()==0 else rets_adj[take].mean())
    sharpe_tr  = float(np.nan if take.sum()==0 else _sharpe(rets_adj[take]))

    # Latest (most recent big-move row)
    latest_X = X.iloc[[-1]]
    latest_p = float(model.predict_proba(latest_X)[0,1])
    latest_sig = "BUY" if latest_p >= HI else "SELL" if latest_p <= LO else "HOLD"

    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "samples_big": int(len(X)),
        "acc_big": float(acc_big),
        "signals_taken": int(take.sum()),
        "signals_rate": float(take.mean()),
        "acc_on_trades": float(acc_tr) if take.sum()>0 else None,
        "precision_buy": float(prec_up) if take.sum()>0 else None,
        "precision_sell": float(prec_dn) if take.sum()>0 else None,
        "mean_trade_after_fees": mean_trade if take.sum()>0 else None,
        "sharpe_trades": sharpe_tr if take.sum()>0 else None,
        "latest_price": float(df["Close"].iloc[-1]),
        "latest_p_up": latest_p,
        "latest_signal": latest_sig,
    }

# ========= main =========
def main():
    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True})
    results = []
    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv_df(ex, sym)
            if df.empty:
                results.append({"symbol": sym, "error": "no data"})
                continue
            r = fit_and_signal(df, sym)
            results.append(r)
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    # Make a tidy table
    cols = ["symbol","rows","samples_big","acc_big","signals_taken","signals_rate",
            "acc_on_trades","precision_buy","precision_sell",
            "mean_trade_after_fees","sharpe_trades","latest_price","latest_p_up","latest_signal","error"]
    tbl = pd.DataFrame([{k: res.get(k) for k in cols} for res in results])

    # Rank coins you might prefer to act on now:
    # 1) latest_signal not HOLD
    # 2) higher latest_p_up (farther from 0.5)
    # 3) historical acc_on_trades, then sharpe_trades
    act_now = tbl.copy()
    act_now["conf"] = (act_now["latest_p_up"] - 0.5).abs()
    act_now = act_now.sort_values(
        by=["latest_signal","conf","acc_on_trades","sharpe_trades"],
        ascending=[True, False, False, False]
    )

    # Print summary
    pd.set_option("display.width", 200)
    print("\n=== Hourly Big-Move Signals (confident only) ===")
    print(tbl[["symbol","acc_big","signals_taken","acc_on_trades","precision_buy","precision_sell",
               "mean_trade_after_fees","sharpe_trades","latest_price","latest_p_up","latest_signal","error"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    print("\n=== Suggested priority (higher confidence & historical quality) ===")
    print(act_now[["symbol","latest_signal","latest_p_up","acc_on_trades","sharpe_trades"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    # Save CSV
    tbl.to_csv("hour_signals_summary.csv", index=False)
    print("\nSaved: hour_signals_summary.csv")

if __name__ == "__main__":
    main()



    
    
