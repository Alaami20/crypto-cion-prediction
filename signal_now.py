import argparse, numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------- indicators ----------
def rsi(s, w=14):
    d = s.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/w, adjust=False).mean()
    rd = dn.ewm(alpha=1/w, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def macd(s): m = ema(s,12) - ema(s,26); return m, ema(m,9)
def sma(s, w): return s.rolling(w).mean()

# ---------- robust Yahoo loader ----------
MAP = {"BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","BNB":"BNB-USD",
       "XRP":"XRP-USD","ADA":"ADA-USD","DOGE":"DOGE-USD","MATIC":"MATIC-USD",
       "DOT":"DOT-USD","LTC":"LTC-USD","LINK":"LINK-USD"}

def norm_ticker(user):
    s = user.strip().upper().replace(" ", "")
    if "/" in s:   # e.g., SOL/USDT -> SOL-USD
        base, quote = s.split("/", 1)
        if quote in ("USDT","USD"): return f"{base}-USD"
    if s in MAP: return MAP[s]
    if not s.endswith("-USD"): s = s + "-USD"
    return s

def load_yf(ticker, start="2018-01-01", interval="1h"):
    df = yf.download(ticker, start=start, interval=interval, progress=False, group_by="column")
    if df.empty: raise RuntimeError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        # if grouped by ticker, pick that slice; else flatten
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1).copy()
        else:
            df.columns = ['_'.join([str(c) for c in t if c]).strip('_') for t in df.columns.to_flat_index()]
    df = df.reset_index()

    # flexible column picking
    cols = {c.lower(): c for c in df.columns}
    def pick(name):
        n = name.lower()
        if n in cols: return cols[n]
        for c in df.columns:
            cl = c.lower()
            if cl.endswith(n) or cl.startswith(n) or f"_{n}" in cl or f"{n}_" in cl:
                return c
        return None
    date = pick("date") or pick("datetime")
    o = pick("open"); h = pick("high"); l = pick("low"); c = pick("close"); v = pick("volume")
    miss = [n for n,v in [("Date",date),("Open",o),("High",h),("Low",l),("Close",c),("Volume",v)] if v is None]
    if miss: raise RuntimeError(f"Missing columns {miss} for {ticker}")

    df = df[[date,o,h,l,c,v]].rename(columns={date:"Date",o:"Open",h:"High",l:"Low",c:"Close",v:"Volume"}).dropna()
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    # drop the very last row if it’s a still-forming bar
    # (yfinance usually gives only closed hourly bars, but this is defensive)
    if len(df) >= 2 and df["Date"].iloc[-1].floor("H") == df["Date"].iloc[-2].floor("H"):
        df = df.iloc[:-1]
    return df

# ---------- features ----------
def make_features(df):
    x = df.copy()
    x["ret_1"] = x["Close"].pct_change()
    x["ret_3"] = x["Close"].pct_change(3)
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_48"] = x["ret_1"].rolling(48).std()
    x["rsi_14"] = rsi(x["Close"], 14)
    x["macd"], x["macd_sig"] = macd(x["Close"])
    x["sma_20"] = sma(x["Close"], 20)
    x["sma_50"] = sma(x["Close"], 50)
    x["sma_200"] = sma(x["Close"], 200)
    x["sma_ratio"] = x["sma_20"] / (x["sma_50"] + 1e-12)
    for k in [1,2,3,6]:
        x[f"ret_lag{k}"] = x["ret_1"].shift(k)

    # next-bar label (up vs down)
    x["fut_ret_1"] = x["Close"].shift(-1) / x["Close"] - 1.0
    x["y"] = (x["fut_ret_1"] > 0).astype(int)

    x = x.dropna().reset_index(drop=True)
    feats = ["ret_1","ret_3","vol_10","vol_48","rsi_14","macd","macd_sig","sma_ratio",
             "ret_lag1","ret_lag2","ret_lag3","ret_lag6"]
    X = x[feats].select_dtypes(include=[np.number]).copy()
    y = x["y"].astype(int)
    return x, X, y

# ---------- decision layer (BUY/HOLD) ----------
def decide(row_last, p_up, hi=0.60):
    """Require both ML confidence and rule confirmation to BUY; else HOLD."""
    rsi_now = float(row_last["rsi_14"])
    macd_now = float(row_last["macd"])
    macd_sig = float(row_last["macd_sig"])
    price    = float(row_last["Close"])
    sma50    = float(row_last["sma_50"])
    sma200   = float(row_last["sma_200"])

    trend_ok = (price > sma50) and (sma50 > sma200)   # uptrend filter
    macd_ok  = macd_now > macd_sig                    # momentum confirm
    rsi_ok   = 30 <= rsi_now <= 65                    # not overbought

    buy_ml   = p_up >= hi
    buy      = buy_ml and trend_ok and macd_ok and rsi_ok

    reasons = []
    if buy_ml: reasons.append(f"ML p_up≥{hi}")
    if trend_ok: reasons.append("Uptrend (Close>SMA50>SMA200)")
    if macd_ok: reasons.append("MACD>Signal")
    if rsi_ok: reasons.append("RSI in [30,65]")

    return ("BUY" if buy else "HOLD"), reasons

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", required=True, help="e.g., btc, eth, sol, or full like BTC-USD / SOL-USD / SOL/USDT")
    ap.add_argument("--tf", default="1h", help="interval: 1h (default). yfinance supports 1h, 30m, 15m, etc (limited history).")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--hi", type=float, default=0.60, help="probability threshold for BUY (default 0.60)")
    args = ap.parse_args()

    ticker = norm_ticker(args.coin)
    df = load_yf(ticker, start=args.start, interval=args.tf)
    if len(df) < 400:
        raise RuntimeError(f"Not enough rows for {ticker} at {args.tf}. Try earlier --start or a longer interval.")

    data, X, y = make_features(df)

    # time-based split
    split = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)

    # quality (on last 20% for reference)
    proba_te = model.predict_proba(X_te)[:,1]
    pred_te  = (proba_te >= 0.5).astype(int)
    try:
        acc = accuracy_score(y_te, pred_te); auc = roc_auc_score(y_te, proba_te)
    except Exception:
        acc, auc = np.nan, np.nan

    # latest bar decision
    latest_X = X.iloc[[-1]]
    latest_row = data.iloc[-1]
    p_up = float(model.predict_proba(latest_X)[0,1])
    action, reasons = decide(latest_row, p_up, hi=args.hi)

    print(f"\n=== {ticker} | interval={args.tf} ===")
    print("Last close:", str(latest_row['Date'])[:19], "| Price:", round(float(latest_row['Close']), 4))
    print(f"Model P(up next bar): {p_up:.3f}  |  Test ACC: {acc:.3f}  AUC: {auc:.3f}")
    print("Decision:", action)
    print("Reasons:", "; ".join(reasons) if reasons else "No strong confirmation (HOLD).")

if __name__ == "__main__":
    main()
