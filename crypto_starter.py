# crypto_starter.py — robust loader, 16-bit features, safe OOF handling
import numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
plt.rcParams["figure.figsize"] = (11, 5)

# ---------- helpers ----------
def _to_series_1d(x)->pd.Series:
    if isinstance(x, pd.Series): return pd.to_numeric(x, errors="coerce")
    if isinstance(x, pd.DataFrame): return pd.to_numeric(x.iloc[:,0], errors="coerce")
    return pd.Series(np.asarray(x).reshape(-1))

def rsi(s,w=14):
    s=_to_series_1d(s); d=s.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
    ru=up.ewm(alpha=1/w, adjust=False).mean(); rd=dn.ewm(alpha=1/w, adjust=False).mean()
    rs=ru/(rd+1e-12); return 100 - (100/(1+rs))

def ema(s, span): s=_to_series_1d(s); return s.ewm(span=span, adjust=False).mean()
def macd(s, f=12, sl=26, sg=9):
    s=_to_series_1d(s); m=ema(s,f)-ema(s,sl); return m, ema(m, sg)
def sma(s,w): s=_to_series_1d(s); return s.rolling(w).mean()
def bollinger(s,w=20,n=2.0):
    s=_to_series_1d(s); mid=sma(s,w); sd=s.rolling(w).std(); return mid+n*sd, mid-n*sd
def sharpe(r, periods=252):
    r=r.dropna(); return np.nan if r.std()==0 else np.sqrt(periods)*r.mean()/r.std()

# ---------- 1) robust download ----------
ticker="BTC-USD"; start_date="2016-01-01"
df = yf.download(ticker, start=start_date, interval="1d",
                 auto_adjust=False, progress=False, group_by="ticker")
if df.empty: raise RuntimeError("yfinance returned no data.")

# unwrap possible MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    if ticker in df.columns.get_level_values(1):
        df = df.xs(ticker, axis=1, level=1).copy()
    else:
        df.columns = ['_'.join([str(c) for c in tup if c]).strip('_')
                      for tup in df.columns.to_flat_index()]

df = df.reset_index()
# map flexible column names
cols_lower = {c.lower(): c for c in df.columns}
def pick(name):
    t=name.lower()
    if t in cols_lower: return cols_lower[t]
    for c in df.columns:
        cl=c.lower()
        if cl.endswith(t) or cl.startswith(t) or f" {t}" in cl or f"_{t}" in cl or f"{t}_" in cl:
            return c
    return None
date_col=pick("date") or pick("datetime")
open_col=pick("open"); high_col=pick("high"); low_col=pick("low")
close_col=pick("close"); vol_col=pick("volume")
missing=[n for n,v in [("Date",date_col),("Open",open_col),("High",high_col),
                       ("Low",low_col),("Close",close_col),("Volume",vol_col)] if v is None]
if missing: raise RuntimeError(f"Missing columns: {missing}\nGot: {list(df.columns)}")

df = df[[date_col,open_col,high_col,low_col,close_col,vol_col]].rename(
    columns={date_col:"Date", open_col:"Open", high_col:"High", low_col:"Low",
             close_col:"Close", vol_col:"Volume"}).dropna()
df = df.sort_values("Date").reset_index(drop=True)
for c in ["Open","High","Low","Close","Volume"]:
    df[c]=pd.to_numeric(df[c], errors="coerce")

print(f"Downloaded {len(df)} rows for {ticker}")

# ---------- 2) features (ratios to avoid float16 overflow) ----------
df["ret_1d"]=df["Close"].pct_change()
df["ret_5d"]=df["Close"].pct_change(5)
df["vol_10"]=df["ret_1d"].rolling(10).std()
df["vol_20"]=df["ret_1d"].rolling(20).std()

df["rsi_14"]=rsi(df["Close"],14)
df["macd"],df["macd_sig"]=macd(df["Close"],12,26,9)

bbh,bbl=bollinger(df["Close"],20,2.0)
close_s=_to_series_1d(df["Close"]).reindex(df.index)
bbh=_to_series_1d(bbh).reindex(df.index); bbl=_to_series_1d(bbl).reindex(df.index)
den=(bbh-bbl).replace(0,np.nan)
df["bb_pos"]=pd.Series((close_s.to_numpy()-bbl.to_numpy())/den.to_numpy(), index=df.index)

# use RELATIVE bands to avoid large magnitudes
df["bb_high_rel"]=bbh/(close_s+1e-12)
df["bb_low_rel"] =bbl/(close_s+1e-12)

df["sma_20"]=sma(df["Close"],20)
df["sma_50"]=sma(df["Close"],50)
df["sma_ratio"]=df["sma_20"]/(df["sma_50"]+1e-12)

df["vol_norm_20"]=df["Volume"]/(df["Volume"].rolling(20).mean()+1e-12)
for k in [1,2,3,5]: df[f"ret_lag{k}"]=df["ret_1d"].shift(k)

# ---------- 3) label ----------
df["y"]=(df["Close"].shift(-1)>df["Close"]).astype(np.int8)

# ---------- 4) dataset (numeric-only, cast to <=16-bit) ----------
exclude={"Date","Open","High","Low","Close","Volume","y","sma_20","sma_50"}  # drop raw big-scale SMAs
feat_cols=[c for c in df.columns if c not in exclude]
data=df.dropna().reset_index(drop=True)

X=data[feat_cols].select_dtypes(include=[np.number]).copy()
# down-cast to float16 safely (ratios/returns are small; should not overflow)
for c in X.columns: X[c]=pd.to_numeric(X[c], errors="coerce").astype(np.float16)
y=data["y"].astype(np.int8)

print(f"Final feature list ({len(X.columns)}): {list(X.columns)}")
print("Data shape:", X.shape)

if len(data)<400: raise RuntimeError("Not enough rows after indicators. Try earlier start_date.")

# ---------- 5) CV + model (no strict assert) ----------
tscv=TimeSeriesSplit(n_splits=5)
oof=np.full(len(X), np.nan, dtype=np.float32)  # allow NaNs for the first chunk (never validated)

for i,(tr,va) in enumerate(tscv.split(X)):
    X_tr=X.iloc[tr].to_numpy(dtype=np.float16); y_tr=y.iloc[tr].to_numpy()
    X_va=X.iloc[va].to_numpy(dtype=np.float16); y_va=y.iloc[va].to_numpy()
    model=HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                         max_iter=400, l2_regularization=1.0,
                                         random_state=42)
    model.fit(X_tr,y_tr)
    proba=model.predict_proba(X_va)[:,1].astype(np.float32)
    pred=(proba>=0.5).astype(np.int8)
    print(f"Fold {i+1}: ACC={accuracy_score(y_va,pred):.3f} | AUC={roc_auc_score(y_va,proba):.3f}")
    oof[va]=proba  # leave earlier training-only chunk as NaN

# ---------- 6) backtest (handle NaNs in proba) ----------
bt=data.copy()
bt["proba"]=oof
bt["signal"]=(bt["proba"]>=0.5).astype("float32").fillna(0).astype(np.int8)

fee=0.0005
bt["next_ret"]=bt["Close"].pct_change().shift(-1)
bt["position"]=bt["signal"]
bt["trade"]=bt["position"].diff().abs().fillna(0)
bt["strategy_ret"]=bt["position"]*bt["next_ret"]-bt["trade"]*fee

strategy=(1+bt["strategy_ret"].fillna(0)).cumprod()
buyhold=(1+bt["next_ret"].fillna(0)).cumprod()
print(f"Final equity (Strategy): {strategy.iloc[-2]:.3f}")
print(f"Final equity (Buy&Hold): {buyhold.iloc[-2]:.3f}")
print(f"Sharpe (approx): {sharpe(bt['strategy_ret']):.2f}")

ax=strategy.plot(label="Strategy"); buyhold.plot(ax=ax,label="Buy & Hold")
ax.set_title(f"Equity Curve (OOF CV) — {ticker}"); ax.legend(); plt.tight_layout(); plt.show()

# Export raw and feature database
df.to_csv("raw_crypto_data.csv", index=False)
data.to_csv("features_crypto_data.csv", index=False)

print("Saved raw_crypto_data.csv and features_crypto_data.csv")
