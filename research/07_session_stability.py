import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
pd.set_option('display.width',250)
def tstat(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 and x.std()>0 else np.nan

print("="*115)
print("GOLD H4 'first bar of new session' (22:00 summer / 23:00 winter UTC) — body return by year")
print("="*115)
d=data.load('xauusd','240')
body=np.log(d.close/d.open)*1e4
sess=d.index.hour.isin([22,23])
t=pd.DataFrame(dict(y=d.index.year, b=body, sess=sess)).dropna()
a=t[t.sess].groupby('y')['b'].agg(n='size', mean='mean', t=tstat).round(2)
b=t[~t.sess].groupby('y')['b'].agg(n='size', mean='mean', t=tstat).round(2)
print("SESSION-OPEN bar:"); print(a.T.to_string())
print("\nALL OTHER bars:"); print(b.T.to_string())
x=t[t.sess]['b']; print(f"\npooled session-open: n={len(x)} mean={x.mean():.2f}bps t={tstat(x):.2f}")
x2=t[~t.sess]['b']; print(f"pooled other bars  : n={len(x2)} mean={x2.mean():.2f}bps t={tstat(x2):.2f}")
print(f"positive years: {(a['mean']>0).sum()}/{len(a)}")

print("\n"+"="*115)
print("Same test on INDICES H4 (session-open bar) and on other symbols")
print("="*115)
rows=[]
for s in data.SYMBOLS:
    d=data.load(s,'240'); body=np.log(d.close/d.open)*1e4
    hrs=sorted(d.index.hour.unique())
    # session-open bar = the bar right after the daily maintenance break
    dt=d.index.to_series().diff().dt.total_seconds()/3600.0
    isopen=(dt>4.0)
    for nm,m in [("session-open", isopen.to_numpy()), ("other", ~isopen.to_numpy())]:
        x=body[m].dropna()
        rows.append(dict(sym=s, bar=nm, n=len(x), bps=x.mean(), t=tstat(x)))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n"+"="*115)
print("GOLD: which H1 hours carry the drift? (H1 2023+, body returns, ann.%)")
print("="*115)
d=data.load('xauusd','60'); body=np.log(d.close/d.open)
t=pd.DataFrame(dict(h=d.index.hour, y=d.index.year, b=body)).dropna()
p=t.groupby(['y','h'])['b'].mean().unstack()*1e4
print((p).round(2).to_string())
