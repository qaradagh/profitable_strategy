"""Quantify the two candidate edges with conditional-return tables."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
pd.set_option('display.width', 240)

def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 and x.std()>0 else np.nan

print("="*110)
print("A) INDEX MEAN REVERSION: forward 1..5d return conditioned on n-day drop, split by 200d trend filter")
print("="*110)
for s in data.INDICES:
    d = data.load(s,'1D'); c = d.close
    r = np.log(c).diff()
    ma200 = c.rolling(200).mean()
    up = c > ma200
    out=[]
    for look in [1,2,3]:
        cum = np.log(c/c.shift(look))
        for cond_name, cond in [(f"drop{look}d", cum<0), ("all", pd.Series(True, index=c.index))]:
            for h in [1,3,5]:
                fwd = np.log(c.shift(-h)/c)
                for trend_name, tcond in [("above200", up), ("below200", ~up)]:
                    m = cond & tcond & ma200.notna() & fwd.notna()
                    x = fwd[m]
                    if len(x) < 60: continue
                    out.append(dict(look=look, cond=cond_name, trend=trend_name, h=h, n=len(x),
                                    mean_bps=x.mean()*1e4, t=tstat(x), hit=(x>0).mean()))
    o=pd.DataFrame(out)
    print(f"\n--- {s.upper()} (D1, {c.index[0].date()} -> {c.index[-1].date()}) ---")
    print(o[o.cond!="all"].to_string(index=False))
    print("baseline(all):")
    print(o[o.cond=="all"].drop_duplicates(subset=['h','trend']).to_string(index=False))

print(); print("="*110)
print("B) RSI(2) style: forward 1..5d return by RSI(2) bucket, above 200MA only")
print("="*110)
def rsi(c, n):
    d = c.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1/n, adjust=False).mean(); ad = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+au/ad.replace(0,np.nan))
for s in data.INDICES + ['xauusd']:
    d=data.load(s,'1D'); c=d.close
    R=rsi(c,2); ma=c.rolling(200).mean(); up=c>ma
    b=pd.cut(R,[0,5,10,20,40,60,80,90,95,100])
    fwd={h: np.log(c.shift(-h)/c) for h in [1,3,5]}
    rows=[]
    for k,g in R.groupby(b, observed=True):
        m = R.index.isin(g.index) & up.values & ma.notna().values
        m = pd.Series(m, index=c.index)
        row=dict(sym=s, bucket=str(k), n=int(m.sum()))
        for h in [1,3,5]:
            x=fwd[h][m].dropna(); row[f'r{h}_bps']=x.mean()*1e4; row[f't{h}']=tstat(x)
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False)); print()

print("="*110)
print("C) TREND FOLLOWING: long-only / long-short MA crossover Sharpe grid (D1, no costs yet)")
print("="*110)
for s in data.SYMBOLS:
    d=data.load(s,'1D'); c=d.close; r=np.log(c).diff().shift(-1)  # next-bar return
    rows=[]
    for fast in [1,5,10,20]:
        for slow in [20,50,100,150,200]:
            if fast>=slow: continue
            sig = (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)
            for mode,pos in [("LO",sig),("LS",2*sig-1)]:
                pnl=(pos*r).dropna()
                rows.append(dict(mode=mode,fast=fast,slow=slow,
                                 sharpe=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else np.nan))
    t=pd.DataFrame(rows)
    piv=t.pivot_table(index=['mode','fast'],columns='slow',values='sharpe')
    print(f"\n--- {s.upper()} ({c.index[0].date()}->{c.index[-1].date()}) buy&hold SR="
          f"{np.log(c).diff().mean()/np.log(c).diff().std()*np.sqrt(252):.2f} ---")
    print(piv.round(2).to_string())
