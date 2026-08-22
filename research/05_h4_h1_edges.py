import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
from strategy.indicators import rsi, zscore
pd.set_option('display.width',240)
def tstat(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 else np.nan

print("=== H4 bar timestamps present, by symbol ===")
for s in ['us500','xauusd','eurusd']:
    d=data.load(s,'240'); print(s, sorted(d.index.hour.unique()), len(d), d.index[0].date())

print("\n"+"="*110)
print("A) H4 MEAN REVERSION: fwd open-to-open return by RSI(2) bucket, close>SMA(200 H4 bars)")
print("="*110)
for s in ['us500','us100','us30','xauusd']:
    d=data.load(s,'240'); c=d.close; o=d.open
    R=rsi(c,2); ma=c.rolling(200).mean(); up=(c>ma)
    rows=[]
    for lo,hi in [(0,5),(5,10),(10,20),(20,35),(35,65),(65,80),(80,90),(90,95),(95,100)]:
        m=(R>lo)&(R<=hi)&up&ma.notna()
        row=dict(sym=s,bucket=f"({lo},{hi}]",n=int(m.sum()))
        for h in [1,3,6,12]:
            fwd=np.log(o.shift(-1-h)/o.shift(-1))   # enter next open, hold h bars
            x=fwd[m].dropna(); row[f'h{h}']=x.mean()*1e4; row[f't{h}']=tstat(x)
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False)); print()

print("="*110)
print("B) H4 TIME-OF-DAY: mean open-to-open return (bps) of the NEXT bar, by bar hour")
print("="*110)
tab={}
for s in ['us500','us100','us30','xauusd','eurusd']:
    d=data.load(s,'240'); o=d.open
    fwd=np.log(o.shift(-1)/o)
    g=fwd.groupby(d.index.hour)
    tab[s]=g.mean()*1e4
    tab[s+'_t']=g.apply(tstat)
print(pd.DataFrame(tab).round(2).to_string())

print("\n"+"="*110)
print("C) H1 SESSION EFFECT on indices (2023+): cumulative ann.return by holding a single hour long")
print("="*110)
rows=[]
for s in ['us500','us100','us30','xauusd']:
    d=data.load(s,'60'); o=d.open
    fwd=np.log(o.shift(-1)/o)     # hold from this bar's open to next bar's open
    for h in range(24):
        m=d.index.hour==h
        x=fwd[m].dropna()
        if len(x)<100: continue
        rows.append(dict(sym=s,hour=h,n=len(x),ann_bps=x.mean()*1e4*252, t=tstat(x)))
t=pd.DataFrame(rows)
print(t.pivot_table(index='hour',columns='sym',values='ann_bps').round(0).to_string())
print("\nt-stats:")
print(t.pivot_table(index='hour',columns='sym',values='t').round(2).to_string())
