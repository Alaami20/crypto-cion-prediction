# High-precision "big move" model for ETH-USD
# Acts only when probability is confident (>= HI or <= LO)
# Prints accuracy on trades only + saves a CSV of signals.

import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

# --------- knobs (tune these) ----------
TICKER = "ETH-USD"
START  = "2016-01-01"
H      = 1        # horizon in days (t -> t+H)
K      = 0.01     # big-move threshold (1% over H days)
HI     = 0.70     # buy if P(up) >= HI
LO     = 0.30     # sell if P(up) <= LO
FEE    = 0.0005   # 5 bps per side (adjust to your venue)

# --------- indicators ----------
def rsi(s, w=14):
    d = s.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/w, adjust=False).mean()
    rd = dn.ewm(alpha=1/w, adjust=False).mean()
    rs = ru/(rd+1e-12); return 100 - (100/(1+rs))
def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def macd(s): m = ema(s,12) - ema(s,26); return m, ema(m,9)
def sma(s, w): return s.rolling(w).mean()
def sharpe(x, periods=252):
    x = pd.Series(x).replace([np.inf,-np.inf], np.nan).dropna()
    return np.nan if x.std()==0 else np.sqrt(periods)*x.mean()/x.std()

# --------- robust download (handles MultiIndex/odd names) ----------
df = yf.download(TICKER, start=START, interval="1d", progress=False, group_by="column")
if df.empty: raise RuntimeError("No data from yfinance.")
if isinstance(df.columns, pd.MultiIndex):
    # unwrap if grouped by ticker level
    tl = df.columns.get_level_values(-1)
    if TICKER in tl: df = df.xs(TICKER, axis=1, level=-1).copy()
    else: df.columns = ['_'.join([str(c) for c in t if c]).strip('_') for t in df.columns.to_flat_index()]
df = df.reset_index()
# pick flexible names
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
for miss,val in [("Date",date),("Open",o),("High",h),("Low",l),("Close",c),("Volume",v)]:
    if val is None: raise RuntimeError(f"Missing {miss} column for {TICKER}")
df = df[[date,o,h,l,c,v]].rename(columns={date:"Date",o:"Open",h:"High",l:"Low",c:"Close",v:"Volume"})
df = df.dropna().sort_values("Date").reset_index(drop=True)
for col in ["Open","High","Low","Close","Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

# --------- features ----------
df["ret_1d"] = df["Close"].pct_change()
df["ret_5d"] = df["Close"].pct_change(5)
df["vol_10"] = df["ret_1d"].rolling(10).std()
df["vol_20"] = df["ret_1d"].rolling(20).std()
df["rsi"] = rsi(df["Close"])
df["macd"], df["macd_sig"] = macd(df["Close"])
df["sma20"] = sma(df["Close"],20)
df["sma50"] = sma(df["Close"],50)
df["sma_ratio"] = df["sma20"] / (df["sma50"] + 1e-12)
for k in [1,2,3,5]:
    df[f"ret_lag{k}"] = df["ret_1d"].shift(k)

# --------- labels: only big moves ----------
df["fut_ret_H"] = df["Close"].shift(-H) / df["Close"] - 1.0
mask_big = df["fut_ret_H"].abs() >= K
df_big = df.loc[mask_big].copy()
df_big["y"] = (df_big["fut_ret_H"] > 0).astype(int)

feat_cols = ["ret_1d","ret_5d","vol_10","vol_20","rsi","macd","macd_sig","sma_ratio","ret_lag1","ret_lag2","ret_lag3","ret_lag5"]
X = df_big[feat_cols].select_dtypes(include=[np.number]).copy()
y = df_big["y"].astype(int)

# (optional) light downcast to float32 for speed/memory
for col in X.columns: X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

# --------- time split (no shuffle) ----------
split = int(len(X) * 0.8)
X_tr, y_tr = X.iloc[:split], y.iloc[:split]
X_te, y_te = X.iloc[split:], y.iloc[split:]
dates_te   = df_big["Date"].iloc[split:].reset_index(drop=True)
futret_te  = df_big["fut_ret_H"].iloc[split:].reset_index(drop=True)
close_te   = df_big["Close"].iloc[split:].reset_index(drop=True)

# --------- model ----------
model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=500, random_state=42)
model.fit(X_tr, y_tr)
proba_te = model.predict_proba(X_te)[:,1]
pred_te  = (proba_te >= 0.5).astype(int)

print(f"[{TICKER}] Big-move test ACC (on big-move days): {accuracy_score(y_te, pred_te):.3f}")

# confident signals only
sig = np.zeros_like(proba_te, dtype=int)   # 1=BUY, -1=SELL, 0=HOLD
sig[proba_te >= HI] = 1
sig[proba_te <= LO] = -1
take = sig != 0

# accuracy/precision on trades only
if take.sum() > 0:
    pred_on_trades = np.where(sig[take]==1, 1, 0)
    y_on_trades    = y_te[take].to_numpy()
    acc_trades     = accuracy_score(y_on_trades, pred_on_trades)
    prec_up        = precision_score(y_on_trades, pred_on_trades, pos_label=1)
    prec_down      = precision_score(1 - y_on_trades, 1 - pred_on_trades, pos_label=1)
else:
    acc_trades = prec_up = prec_down = np.nan

# simple trade PnL (enter at close, exit after H days)
rets = np.zeros_like(proba_te, dtype=float)
rets[sig==1]  = futret_te[sig==1].to_numpy()
rets[sig==-1] = -futret_te[sig==-1].to_numpy()
# fee for entry & exit
rets_adj = rets - 2*FEE*(sig!=0).astype(float)
equity = (1 + pd.Series(rets_adj)).replace([np.inf,-np.inf], np.nan).fillna(0).add(1).cumprod()

print(f"Signals taken (confident): {int(take.sum())} ({take.mean():.1%} of test)")
if take.sum()>0:
    print(f"Accuracy on trades only: {acc_trades:.3f}")
    print(f"Precision BUY:  {prec_up:.3f}")
    print(f"Precision SELL: {prec_down:.3f}")
    print(f"Mean trade return after fees: {rets_adj[take].mean():.4f}")
    print(f"Sharpe (approx, trades only): {sharpe(rets_adj[take]):.2f}")

# latest signal (on the most recent big-move sample)
latest_X = X.iloc[[-1]]
latest_p = float(model.predict_proba(latest_X)[0,1])
latest_sig = "BUY" if latest_p >= HI else "SELL" if latest_p <= LO else "HOLD"

print("\n--- Latest ---")
print("Date:", str(df_big["Date"].iloc[-1])[:10])
print("Close:", round(float(df_big["Close"].iloc[-1]), 2))
print("P(up):", round(latest_p,3), "| Signal:", latest_sig)

# --------- export signals CSV ---------
out = pd.DataFrame({
    "date": dates_te,
    "close": close_te,
    "y_true_up": y_te.reset_index(drop=True),
    "p_up": proba_te,
    "signal": sig,     # 1=BUY, -1=SELL, 0=HOLD
    "fut_ret_H": futret_te,
    "trade_ret_after_fees": rets_adj
})
out.to_csv("eth_signals.csv", index=False)
print("\nSaved: eth_signals.csv (test set signals & PnL)")
