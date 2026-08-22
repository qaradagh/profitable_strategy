"""Is the 22:00/23:00 UTC 'session open' edge real, or a rollover print artifact?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
pd.set_option('display.width',240)
def tstat(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 else np.nan

for s in ['xauusd','us500']:
    print("="*100); print(f"{s.upper()} H1 — behaviour around the daily rollover"); print("="*100)
    d=data.load(s,'60')
    gap = np.log(d.open/d.close.shift(1))*1e4     # gap between consecutive bars
    body= np.log(d.close/d.open)*1e4              # the bar's own move
    rng = np.log(d.high/d.low)*1e4                # bar range
    t=pd.DataFrame(dict(hour=d.index.hour, gap=gap, body=body, rng=rng)).dropna()
    g=t.groupby('hour').agg(n=('gap','size'), gap_bps=('gap','mean'), gap_t=('gap',tstat),
                            body_bps=('body','mean'), body_t=('body',tstat), range_bps=('rng','mean'))
    print(g.round(2).to_string())
    print()

print("="*100); print("Raw XAUUSD H1 bars across a rollover (sample days)"); print("="*100)
d=data.load('xauusd','60')
for day in ['2024-03-12','2025-06-10','2026-02-04']:
    w=d.loc[f'{day} 18:00':f'{day} 23:59']
    nxt=d.loc[f'{day} 23:59':].head(3)
    print(pd.concat([w,nxt]).to_string()); print()

print("="*100); print("Do consecutive H1 bars chain (open[t] == close[t-1])?"); print("="*100)
for s in ['xauusd','us500','eurusd']:
    d=data.load(s,'60')
    same=(d.open==d.close.shift(1))
    t=pd.DataFrame(dict(hour=d.index.hour, same=same)).dropna()
    print(f"{s}: overall chained {same.mean():.1%}")
    print("  by hour:", t.groupby('hour')['same'].mean().round(2).to_dict())
