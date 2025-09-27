import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# --- indicators ---
def rsi(s, w=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/w, adjust=False).mean()
    rd = dn.ewm(alpha=1/w, adjust=False).mean()
    rs = ru/(rd+1e-12); return 100-(100/(1+rs))

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def macd(s): return ema(s,12)-ema(s,26), ema(ema(s,12)-ema(s,26),9)
def sma(s, w): return s.rolling(w).mean()

# --- load data ---
ticker = "BTC-USD"
df = yf.download(ticker, start="2018-01-01", interval="1d", progress=False)
df = df.reset_index()[["Date","Open","High","Low","Close","Volume"]].dropna()

# --- features ---
df["ret_1d"] = df["Close"].pct_change()
df["rsi"] = rsi(df["Close"])
df["macd"], df["macd_sig"] = macd(df["Close"])
df["sma20"] = sma(df["Close"],20)
df["sma50"] = sma(df["Close"],50)
df["sma_ratio"] = df["sma20"]/df["sma50"]

# target: tomorrow up?
df["y"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

X = df[["ret_1d","rsi","macd","macd_sig","sma_ratio"]]
y = df["y"]

# --- train/test ---
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
model = HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
print("Test Accuracy:", accuracy_score(y_test,y_pred))
print("Test AUC:", roc_auc_score(y_test,y_proba))

# --- latest signal ---
latest = X.iloc[[-1]]
proba = model.predict_proba(latest)[0,1]
signal = "BUY" if proba>=0.55 else "SELL" if proba<=0.45 else "HOLD"

# --- rule overlay ---
rsi_now = df["rsi"].iloc[-1]; macd_now = df["macd"].iloc[-1]; macd_sig = df["macd_sig"].iloc[-1]
sma20 = df["sma20"].iloc[-1]; sma50 = df["sma50"].iloc[-1]
price = df["Close"].iloc[-1]

rules = []
if rsi_now < 30 and price > sma50: rules.append("STRONG BUY (RSI oversold)")
if rsi_now > 70 or macd_now < macd_sig: rules.append("STRONG SELL (overbought/MACD down)")

print("\n--- INVESTMENT SIGNAL ---")
print("Date:", df["Date"].iloc[-1].date())
print("Price:", round(price,2))
print("Model probability up tomorrow:", round(proba,3))
print("Model signal:", signal)
if rules: print("Rule-based signals:", "; ".join(rules))
else: print("No strong rule signal")
