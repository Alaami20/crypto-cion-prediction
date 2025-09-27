import sys, numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

MAP = {
    "etr": "ETR",        # Entergy Corp (stock)
    "eth": "ETH-USD",    # Ethereum (crypto)
}

def rsi(s, w=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/w, adjust=False).mean()
    rd = dn.ewm(alpha=1/w, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def macd(s): 
    m = ema(s, 12) - ema(s, 26)
    return m, ema(m, 9)
def sma(s, w): return s.rolling(w).mean()

def load_data(ticker, start="2016-01-01", interval="1d"):
    df = yf.download(ticker, start=start, interval=interval, progress=False, group_by="column")
    if df.empty:
        raise RuntimeError("No data from yfinance for " + ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # if grouped by ticker, pick that slice; else flatten
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1).copy()
        else:
            df.columns = ['_'.join([str(c) for c in t if c]).strip('_') for t in df.columns.to_flat_index()]
    df = df.reset_index()
    # pick flexible column names
    cols = {c.lower(): c for c in df.columns}
    def pick(name):
        n = name.lower()
        if n in cols: return cols[n]
        for c in df.columns:
            cl = c.lower()
            if cl.endswith(n) or cl.startswith(n) or f"_{n}" in cl or f"{n}_" in cl: return c
        return None
    date = pick("date") or pick("datetime")
    o = pick("open"); h = pick("high"); l = pick("low"); c = pick("close"); v = pick("volume")
    for miss, val in [("Date", date), ("Open", o), ("High", h), ("Low", l), ("Close", c), ("Volume", v)]:
        if val is None: raise RuntimeError(f"Missing {miss} column for {ticker}")
    df = df[[date, o, h, l, c, v]].rename(columns={date:"Date", o:"Open", h:"High", l:"Low", c:"Close", v:"Volume"}).dropna()
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().sort_values("Date").reset_index(drop=True)

def build_features(df):
    out = df.copy()
    out["ret_1d"]  = out["Close"].pct_change()
    out["ret_5d"]  = out["Close"].pct_change(5)
    out["vol_10"]  = out["ret_1d"].rolling(10).std()
    out["vol_20"]  = out["ret_1d"].rolling(20).std()
    out["rsi"]     = rsi(out["Close"])
    out["macd"], out["macd_sig"] = macd(out["Close"])
    out["sma20"]   = sma(out["Close"], 20)
    out["sma50"]   = sma(out["Close"], 50)
    out["sma_ratio"] = out["sma20"] / (out["sma50"] + 1e-12)
    out["y"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.dropna().reset_index(drop=True)
    feats = ["ret_1d","ret_5d","vol_10","vol_20","rsi","macd","macd_sig","sma_ratio"]
    X = out[feats].select_dtypes(include=[np.number]).copy()
    y = out["y"].astype(int)
    return out, X, y

def rule_overlay(row):
    rsi_now = row["rsi"]; macd_now = row["macd"]; macd_sig = row["macd_sig"]
    sma_ratio = row["sma_ratio"]; rules = []
    if rsi_now < 30 and sma_ratio > 1.0: rules.append("STRONG BUY (RSI<30 & >SMA50)")
    if rsi_now > 70 or macd_now < macd_sig: rules.append("STRONG SELL (RSI>70/MACD<Signal)")
    return rules

def main():
    key = (sys.argv[1].lower() if len(sys.argv) > 1 else "etr")
    ticker = MAP.get(key, key.upper())  # allow passing any symbol
    df = load_data(ticker)
    data, X, y = build_features(df)
    # split (time order)
    split = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:,1]
    pred  = (proba >= 0.5).astype(int)
    print(f"[{ticker}] Test ACC: {accuracy_score(y_te, pred):.3f} | AUC: {roc_auc_score(y_te, proba):.3f}")

    # latest signal
    last = X.iloc[[-1]]
    p_up = float(model.predict_proba(last)[0,1])
    signal = "BUY" if p_up >= 0.55 else "SELL" if p_up <= 0.45 else "HOLD"
    last_row = data.iloc[-1]
    rules = rule_overlay(last_row)

    print("\n--- INVESTMENT SIGNAL ---")
    print("Ticker:", ticker)
    print("Date:", str(last_row["Date"])[:10])
    print("Close:", round(float(last_row["Close"]), 4))
    print("Prob up (t+1):", round(p_up, 3))
    print("Model signal:", signal)
    if rules:
        print("Rule-based signals:", "; ".join(rules))
    else:
        print("Rule-based signals: None")

    # show last 5 days context
    tail = data[["Date","Close","rsi","macd","macd_sig","sma_ratio"]].tail(5)
    print("\nLast 5 days context:")
    print(tail.to_string(index=False))

if __name__ == "__main__":
    main()
