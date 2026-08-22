"""Hunt for sleeves that diversify the gold-session book."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
from strategy.indicators import rsi
from strategy.strategies.session import session_open_mask
pd.set_option('display.width',250)
def tstat(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 and x.std()>0 else np.nan

print("="*115); print("A) GOLD H4 MOMENTUM — is it independent of the session bar?"); print("="*115)
d=data.load('xauusd','240'); o,c=d.open,d.close
R=rsi(c,2); sess=session_open_mask(d.index)
nxt_sess=np.r_[sess[1:], False]      # bar whose *next* bar is the session bar
fwd1=np.log(o.shift(-2)/o.shift(-1))*1e4     # hold next bar
fwd3=np.log(o.shift(-4)/o.shift(-1))*1e4
rows=[]
for nm,lo,hi in [("RSI<10",0,10),("RSI 10-20",10,20),("RSI 80-90",80,90),("RSI>90",90,100),("RSI>95",95,100)]:
    m=(R>lo)&(R<=hi)
    for sub,mm in [("all",m),("excl. next=session",m&~pd.Series(nxt_sess,index=d.index))]:
        x1=fwd1[mm].dropna(); x3=fwd3[mm].dropna()
        rows.append(dict(bucket=nm,subset=sub,n=len(x1),h1=x1.mean(),t1=tstat(x1),h3=x3.mean(),t3=tstat(x3)))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n"+"="*115); print("B) INDEX MEAN REVERSION on H4 — pooled us500/us100/us30, fwd open-to-open")
print("="*115)
rows=[]
for s in ['us500','us100','us30']:
    d=data.load(s,'240'); o,c=d.open,d.close
    R=rsi(c,2); ma=c.rolling(200).mean(); up=c>ma
    for nm,cond in [("RSI<10 & >MA200",(R<10)&up), ("RSI<20 & >MA200",(R<20)&up),
                    ("RSI<20 (no filter)",R<20), ("RSI<5 & >MA200",(R<5)&up)]:
        for h in [3,6,12,18]:
            fwd=np.log(o.shift(-1-h)/o.shift(-1))*1e4
            x=fwd[cond&ma.notna()].dropna()
            rows.append(dict(sym=s,cond=nm,hold_bars=h,n=len(x),bps=x.mean(),t=tstat(x)))
t=pd.DataFrame(rows)
print(t.pivot_table(index=['cond','hold_bars'],columns='sym',values='bps').round(2).to_string())
print("\nt-stats:")
print(t.pivot_table(index=['cond','hold_bars'],columns='sym',values='t').round(2).to_string())

print("\n"+"="*115); print("C) INDEX intraday session: hold indices only during selected NY hours (H1, 2023+)")
print("="*115)
rows=[]
for s in ['us500','us100','us30']:
    d=data.load(s,'60'); b=np.log(d.close/d.open)*1e4
    ny=d.index.tz_convert('America/New_York')
    t2=pd.DataFrame(dict(h=ny.hour,b=b.values)).dropna()
    g=t2.groupby('h')['b'].agg(n='size',bps='mean',t=tstat)
    g.columns=pd.MultiIndex.from_product([[s],g.columns])
    rows.append(g)
print(pd.concat(rows,axis=1).round(2).to_string())

print("\n"+"="*115); print("D) GOLD: the mirror trade — is the London PM / NY afternoon leg shortable? (H1 2023+, H4 2013+)")
print("="*115)
for tf in ['60','240']:
    d=data.load('xauusd',tf); b=np.log(d.close/d.open)*1e4
    ny=d.index.tz_convert('America/New_York')
    t3=pd.DataFrame(dict(h=ny.hour,b=b.values)).dropna()
    print(f"--- xauusd {tf} ---")
    print(t3.groupby('h')['b'].agg(n='size',bps='mean',t=tstat).round(2).T.to_string())
