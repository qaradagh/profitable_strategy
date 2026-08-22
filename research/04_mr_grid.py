"""In-sample parameter surface for the index MR sleeve. Looking for plateaus."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data, engine, metrics
from strategy.strategies import index_mean_reversion
pd.set_option('display.width',250)

IS_END = '2018-12-31'
SYMS = ['us500','us100','us30']
frames = {s: data.load(s,'1D') for s in SYMS}

def run(params, period):
    nets=[]
    for s in SYMS:
        d=frames[s]
        t=index_mean_reversion(d, **params)
        r=engine.simulate(d,t,s)
        n=r.net
        n = n[:IS_END] if period=='IS' else n[IS_END:]
        if len(n)>50: nets.append((s,n,r))
    if not nets: return None
    # equal-weight the three index sleeves
    df=pd.DataFrame({s:n for s,n,_ in nets}).fillna(0.0)
    tot=df.mean(axis=1)
    st=metrics.stats(tot)
    ntr=sum(len(r.trades) for _,_,r in nets)
    return dict(sharpe=st['sharpe'], cagr=st['cagr'], maxdd=st['maxdd'],
                vol=st['vol'], expo=st['exposure'], trades=ntr, ret=tot)

grid = dict(
    rsi_len=[2,3,4],
    entry=[5,10,15,20,25],
    exit_rsi=[50,60,70,80],
    exit_ma=[0,5,10],
    max_hold=[3,5,8,12],
)
keys=list(grid)
rows=[]
for combo in itertools.product(*[grid[k] for k in keys]):
    p=dict(zip(keys,combo))
    p2=dict(p); 
    if p2['exit_ma']==0: p2['exit_ma']=1_000_000   # effectively disable
    p2.update(trend_len=200, stop=0.06)
    r=run(p2,'IS')
    if r is None or r['trades']<40: continue
    rows.append(dict(**p, **{k:v for k,v in r.items() if k!='ret'}))
g=pd.DataFrame(rows).sort_values('sharpe',ascending=False)
print(f"grid points evaluated: {len(g)}")
print("\n--- TOP 20 IN-SAMPLE (1999->2018, pooled 3 indices, equal weight) ---")
print(g.head(20).round(3).to_string(index=False))

print("\n--- MARGINAL EFFECT of each parameter (mean IS Sharpe) ---")
for k in keys:
    m=g.groupby(k)['sharpe'].agg(['mean','median','max','size']).round(3)
    print(f"\n{k}:\n{m.to_string()}")
g.to_csv('research/_mr_grid_is.csv', index=False)
