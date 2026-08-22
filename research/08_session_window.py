"""Pin down the exact tradeable window for the gold session edge (DST-safe)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
pd.set_option('display.width',250)
def tstat(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 and x.std()>0 else np.nan

def sessionize(d, bar_hours):
    """Label each bar by (session_id, slot) where slot 0 = first bar after break."""
    dt = d.index.to_series().diff().dt.total_seconds()/3600.0
    newsess = (dt > bar_hours*1.5) | dt.isna()
    sid = newsess.cumsum()
    slot = d.groupby(sid).cumcount() if False else pd.Series(np.arange(len(d)), index=d.index).groupby(sid.values).rank(method='first').astype(int)-1
    return sid, slot

print("="*118)
print("A) GOLD H1 (2023+): mean body return (bps) by SLOT after session open")
print("="*118)
d=data.load('xauusd','60'); sid,slot=sessionize(d,1)
body=np.log(d.close/d.open)*1e4
t=pd.DataFrame(dict(slot=slot.values, b=body.values, y=d.index.year)).dropna()
g=t[t.slot<12].groupby('slot')['b'].agg(n='size', bps='mean', t=tstat).round(2)
print(g.T.to_string())
print("\nby year (bps):")
print(t[t.slot<8].pivot_table(index='y',columns='slot',values='b',aggfunc='mean').round(2).to_string())

print("\n"+"="*118)
print("B) GOLD H1 (2023+): CUMULATIVE window returns, enter at slot i, exit at slot j (bps)")
print("="*118)
piv=t.pivot_table(index=t.index, columns='slot', values='b')  # not needed; do direct
# build per-session cumulative
tmp=pd.DataFrame(dict(sid=sid.values, slot=slot.values, b=body.values)).dropna()
mat=tmp.pivot_table(index='sid',columns='slot',values='b')
rows=[]
for i in range(0,8):
    for j in range(i+1,13):
        if j-1 not in mat.columns or i not in mat.columns: continue
        seg=mat.loc[:, i:j-1].sum(axis=1, min_count=(j-i))
        seg=seg.dropna()
        if len(seg)<300: continue
        rows.append(dict(enter_slot=i, hold_h=j-i, n=len(seg), bps=seg.mean(), t=tstat(seg), std=seg.std()))
r=pd.DataFrame(rows)
print(r.pivot_table(index='enter_slot',columns='hold_h',values='bps').round(2).to_string())
print("\nt-stats:")
print(r.pivot_table(index='enter_slot',columns='hold_h',values='t').round(2).to_string())

print("\n"+"="*118)
print("C) GOLD H4 (2013+, LONG HISTORY): slot returns and cumulative windows")
print("="*118)
d4=data.load('xauusd','240'); sid4,slot4=sessionize(d4,4)
b4=np.log(d4.close/d4.open)*1e4
t4=pd.DataFrame(dict(sid=sid4.values, slot=slot4.values, b=b4.values, y=d4.index.year)).dropna()
print(t4[t4.slot<6].groupby('slot')['b'].agg(n='size', bps='mean', t=tstat).round(2).T.to_string())
mat4=t4.pivot_table(index='sid',columns='slot',values='b')
rows=[]
for i in range(0,4):
    for j in range(i+1,7):
        seg=mat4.loc[:, i:j-1].sum(axis=1, min_count=(j-i)).dropna()
        if len(seg)<500: continue
        rows.append(dict(enter=i,hold_bars=j-i,n=len(seg),bps=seg.mean(),t=tstat(seg),std=seg.std()))
r4=pd.DataFrame(rows)
print("\ncumulative bps:"); print(r4.pivot_table(index='enter',columns='hold_bars',values='bps').round(2).to_string())
print("t-stat:");         print(r4.pivot_table(index='enter',columns='hold_bars',values='t').round(2).to_string())
print("std (bps):");      print(r4.pivot_table(index='enter',columns='hold_bars',values='std').round(1).to_string())
print("\nimplied gross annualised Sharpe (260 sessions/yr):")
r4['sharpe']=r4.bps/r4['std']*np.sqrt(260)
print(r4.pivot_table(index='enter',columns='hold_bars',values='sharpe').round(2).to_string())
