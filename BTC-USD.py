# big_move_signals.py
# High-precision "big move" model:
# - Labels only when |return(t→t+H)| >= K (e.g., 1%)
# - Acts only when probability is confident (>= hi or <= lo)
# - Reports accuracy on trades, precision, and trade count

import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

# ---------- knobs you can tweak ----------
TICKER = "BTC-USD"   # or "ETH-USD", "ETR", etc.
START  = "2016-01-01"
H      = 1           # horizon in days (t -> t+H)
K      = 0.01        # big-move threshold (1% over H days)
HI     = 0.70        # buy threshold for P(up)
LO     = 0.30        # sell threshold for P(up)

# ---------- indicators ----------
def rsi(s, w=14):
    d = s.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1/w, adjust=False).mean()
    rd = dn.ewm(alpha=1/w, adjust=False).mean()
    rs = ru/(rd+1e-12); return 100 - (100/(1+rs))

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def macd(s): 
    m = ema(s,12) - ema(s,26)
    return m, ema(m,9)
def sma(s, w): return s.rolling(w).mean()

# ---------- data ----------
df = yf.download(TICKER, start=START, interval="1d", progress=False, group_by="column")
if df.empty: raise RuntimeError("No data from yfinance.")
if isinstance(df.columns, pd.MultiIndex):
    # unwrap if grouped by ticker level
    tl = df.columns.get_level_values(-1)
    if TICKER in tl: df = df.xs(TICKER, axis=1, level=-1).copy()
    else: df.columns = ['_'.join([str(c) for c in t if c]).strip('_') for t in df.columns.to_flat_index()]
df = df.reset_index()[["Date","Open","High","Low","Close","Volume"]].dropna().sort_values("Date").reset_index(drop=True)
for c in ["Open","High","Low","Close","Volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- features ----------
df["ret_1d"] = df["Close"].pct_change()
df["ret_5d"] = df["Close"].pct_change(5)
df["vol_10"] = df["ret_1d"].rolling(10).std()
df["vol_20"] = df["ret_1d"].rolling(20).std()
df["rsi"]    = rsi(df["Close"])
df["macd"], df["macd_sig"] = macd(df["Close"])
df["sma20"]  = sma(df["Close"], 20)
df["sma50"]  = sma(df["Close"], 50)
df["sma_ratio"] = df["sma20"]/(df["sma50"]+1e-12)
for k in [1,2,3,5]: df[f"ret_lag{k}"] = df["ret_1d"].shift(k)

# ---------- labels: only big moves ----------
# future H-day return
df["fut_ret_H"] = df["Close"].shift(-H) / df["Close"] - 1.0
# keep only rows with big absolute move
mask_big = df["fut_ret_H"].abs() >= K
df_big = df.loc[mask_big].copy()

# binary label among big moves: up vs down
df_big["y"] = (df_big["fut_ret_H"] > 0).astype(int)

# build X (numeric features only)
feat_cols = ["ret_1d","ret_5d","vol_10","vol_20","rsi","macd","macd_sig","sma_ratio","ret_lag1","ret_lag2","ret_lag3","ret_lag5"]
X = df_big[feat_cols].select_dtypes(include=[np.number]).copy()
y = df_big["y"].astype(int)

# optional: 16-bit features if you need tight memory
for c in X.columns: X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float16)

# ---------- time split (no shuffle) ----------
split = int(len(X)*0.8)
X_tr, y_tr = X.iloc[:split], y.iloc[:split]
X_te, y_te = X.iloc[split:], y.iloc[split:]
dates_te   = df_big["Date"].iloc[split:]
close_te   = df_big["Close"].iloc[split:]
futret_te  = df_big["fut_ret_H"].iloc[split:]

# ---------- model ----------
model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, random_state=42)
model.fit(X_tr, y_tr)

proba_te = model.predict_proba(X_te)[:,1]
# confident signals only
sig = np.zeros_like(proba_te, dtype=int)   # 1=BUY, -1=SELL, 0=HOLD
sig[proba_te >= HI] = 1
sig[proba_te <= LO] = -1
take = sig != 0

# accuracy / precision on trades only
if take.sum() > 0:
    # true label mapping: up=1, down=0  → for SELL, we “predict 0”
    pred_on_trades = np.where(sig[take]==1, 1, 0)
    y_on_trades    = y_te[take].to_numpy()
    acc_trades     = accuracy_score(y_on_trades, pred_on_trades)
    prec_up        = precision_score(y_on_trades, pred_on_trades, pos_label=1)
    prec_down      = precision_score(1 - y_on_trades, 1 - pred_on_trades, pos_label=1)  # precision for "down"
else:
    acc_trades, prec_up, prec_down = np.nan, np.nan, np.nan

# naive backtest on trades only (enter at close, exit after H days; long for BUY, short for SELL)
rets = np.zeros_like(proba_te, dtype=float)
rets[sig==1]  = futret_te[sig==1].to_numpy()      # long earns the future return
rets[sig==-1] = -futret_te[sig==-1].to_numpy()    # short earns the negative of future return
fee = 0.0005
# simple fee model: pay fee on entry & exit for each trade (approx)
turnover = (sig != 0).astype(int)
rets_adj = rets - 2*fee*turnover
equity = (1 + pd.Series(rets_adj)).replace([np.inf, -np.inf], np.nan).fillna(0).add(1).cumprod()

def sharpe(x, periods=252):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    return np.nan if x.std()==0 else np.sqrt(periods)*x.mean()/x.std()

print(f"\n=== {TICKER} big-move model (H={H} day, K={int(K*100)}% threshold) ===")
print(f"Total samples in test (big-move days only): {len(X_te)}")
print(f"Signals taken (confident): {int(take.sum())} ({take.mean():.1%} of test)")
print(f"Accuracy on trades only: {acc_trades:.3f}" if take.sum()>0 else "No trades at chosen thresholds.")
print(f"Precision BUY:  {prec_up:.3f}"   if take.sum()>0 else "")
print(f"Precision SELL: {prec_down:.3f}" if take.sum()>0 else "")
print(f"Mean trade return (after fees): {rets_adj[take].mean():.4f}" if take.sum()>0 else "")
print(f"Sharpe (approx, trades only):   {sharpe(rets_adj[take]):.2f}" if take.sum()>0 else "")

# latest signal
latest = X.iloc[[-1]]
p_up = float(model.predict_proba(latest)[0,1])
latest_sig = "BUY" if p_up>=HI else "SELL" if p_up<=LO else "HOLD"
print("\n--- Latest ---")
print("Date:", str(df_big['Date'].iloc[-1])[:10])
print("Close:", round(float(df_big['Close'].iloc[-1]), 2))
print("P(up):", round(p_up,3), "| Signal:", latest_sig)
