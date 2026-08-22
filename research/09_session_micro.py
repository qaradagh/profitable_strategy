import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
pd.set_option('display.width',250)
def tstat(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 and x.std()>0 else np.nan

print("="*118); print("A) GOLD 5m (2026-05..08, ~70 sessions): shape of the first 2 hours after session open")
print("="*118)
d=data.load('xauusd','5')
dt=d.index.to_series().diff().dt.total_seconds()/60.0
sid=((dt>5*1.5)|dt.isna()).cumsum()
slot=pd.Series(np.arange(len(d)),index=d.index).groupby(sid.values).rank(method='first').astype(int)-1
b=np.log(d.close/d.open)*1e4
t=pd.DataFrame(dict(sid=sid.values, slot=slot.values, b=b.values)).dropna()
g=t[t.slot<24].groupby('slot')['b'].agg(n='size',bps='mean',t=tstat)
g['cum_from_0']=g.bps.cumsum()
print(g.round(2).T.to_string())

mat=t.pivot_table(index='sid',columns='slot',values='b')
rows=[]
for i in [0,1,2,3,6,12]:
    for j in [i+3,i+6,i+12,i+24,i+36]:
        if j-1 not in mat.columns: continue
        seg=mat.loc[:, i:j-1].sum(axis=1,min_count=(j-i)).dropna()
        if len(seg)<50: continue
        rows.append(dict(enter_min=i*5, hold_min=(j-i)*5, n=len(seg), bps=seg.mean(), t=tstat(seg)))
r=pd.DataFrame(rows)
print("\ncumulative bps by (entry offset, hold):")
print(r.pivot_table(index='enter_min',columns='hold_min',values='bps').round(2).to_string())
print("t-stats:"); print(r.pivot_table(index='enter_min',columns='hold_min',values='t').round(2).to_string())

print("\n"+"="*118); print("B) FILTERS on the gold session trade (H4 session bar, 2013+)")
print("="*118)
d4=data.load('xauusd','240')
sess=d4.index.hour.isin([22,23])
body=np.log(d4.close/d4.open)*1e4
prev_ret = np.log(d4.open/d4.open.shift(1))*1e4          # previous H4 bar move
prev_day = np.log(d4.open/d4.open.shift(6))*1e4          # previous day move
ma200 = d4.close.rolling(200).mean()
above = (d4.open > ma200)
dow = d4.index.dayofweek
X=pd.DataFrame(dict(b=body, prev=prev_ret, pday=prev_day, above=above, dow=dow))[sess].dropna()
print(f"baseline: n={len(X)} bps={X.b.mean():.2f} t={tstat(X.b):.2f}")
for nm, m in [
    ("above 200MA", X.above), ("below 200MA", ~X.above),
    ("prev bar up", X.prev>0), ("prev bar down", X.prev<=0),
    ("prev day up", X.pday>0), ("prev day down", X.pday<=0),
    ("prev day |move|>50bps", X.pday.abs()>50), ("prev day quiet", X.pday.abs()<=50),
]:
    x=X.b[m]; print(f"  {nm:24s} n={len(x):5d} bps={x.mean():6.2f} t={tstat(x):5.2f}  hit={100*(x>0).mean():.1f}%")
print("\nby weekday (0=Mon .. 4=Fri):")
print(X.groupby('dow')['b'].agg(n='size',bps='mean',t=tstat).round(2).to_string())
print("\nby year x above200:")
print(X.assign(y=X.index.year).pivot_table(index='y',columns='above',values='b',aggfunc='mean').round(2).to_string())
