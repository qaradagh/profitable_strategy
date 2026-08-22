import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data, engine, metrics
from strategy.strategies import index_mean_reversion
pd.set_option('display.width',250)

SYMS=['us500','us100','us30']
F={s:data.load(s,'240') for s in SYMS}
IS_END='2022-12-31'

def run(p, period):
    nets={}; ntr=0
    for s in SYMS:
        t=index_mean_reversion(F[s], **p)
        r=engine.simulate(F[s],t,s)
        n=r.net.groupby(r.net.index.normalize()).sum()
        n = n[:IS_END] if period=='IS' else n[IS_END:]
        nets[s]=n; ntr+=((r.trades.entry<=IS_END).sum() if period=='IS' else (r.trades.entry>IS_END).sum())
    tot=pd.DataFrame(nets).fillna(0.0).mean(axis=1)
    st=metrics.stats(tot)
    return dict(sharpe=st['sharpe'],cagr=st['cagr'],maxdd=st['maxdd'],vol=st['vol'],trades=int(ntr)), tot

grid=dict(entry=[10,15,20,25], exit_rsi=[50,60,70], max_hold=[6,12,18,24,30], stop=[0.03,0.05,0.99])
keys=list(grid); rows=[]
for combo in itertools.product(*[grid[k] for k in keys]):
    p=dict(zip(keys,combo)); p.update(rsi_len=2, exit_ma=10**9, trend_len=200)
    a,_=run(p,'IS'); b,_=run(p,'OOS')
    rows.append(dict(**dict(zip(keys,combo)), is_sharpe=a['sharpe'], is_cagr=a['cagr'], is_dd=a['maxdd'],
                     oos_sharpe=b['sharpe'], oos_cagr=b['cagr'], oos_dd=b['maxdd'], is_tr=a['trades'], oos_tr=b['trades']))
g=pd.DataFrame(rows)
print(f"evaluated {len(g)} parameter sets   (IS 2017-2022, OOS 2023-2026)")
print("\n--- TOP 15 BY IN-SAMPLE SHARPE (and what they did out-of-sample) ---")
print(g.sort_values('is_sharpe',ascending=False).head(15).round(3).to_string(index=False))
print("\n--- IS vs OOS rank correlation (is the surface informative or noise?) ---")
print(f"  spearman(is_sharpe, oos_sharpe) = {g.is_sharpe.corr(g.oos_sharpe, method='spearman'):.3f}")
print(f"  mean IS sharpe {g.is_sharpe.mean():.2f} | mean OOS sharpe {g.oos_sharpe.mean():.2f}")
print("\n--- MARGINAL EFFECTS (mean sharpe) ---")
for k in keys:
    print(f"\n{k}:\n{g.groupby(k)[['is_sharpe','oos_sharpe']].mean().round(3).to_string()}")
g.to_csv('research/_idx_h4_grid.csv',index=False)
