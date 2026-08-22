import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data, engine, metrics, costs
from strategy.strategies.session import gold_session
pd.set_option('display.width',250)

d4=data.load('xauusd','240'); d1=data.load('xauusd','60')
print("="*112); print("A) SUNDAY FILTER (now testing the session bar's weekday)"); print("="*112)
rows=[]
for skip in [True,False]:
    for tf,d,hold in [('H4',d4,1),('H1',d1,3)]:
        t=gold_session(d,hold_bars=hold,skip_sunday=skip)
        r=engine.simulate(d,t,'xauusd')
        rows.append(dict(**metrics.summary_row(f'{tf} skipSun={skip}', r.net), trades=len(r.trades)))
print(metrics.fmt(rows)[['strategy','years','cagr','vol','sharpe','maxdd','calmar','trades']].to_string(index=False))

print("\n"+"="*112); print("B) COST STRESS TEST — how wide can the session-open spread get before the edge dies?")
print("="*112)
rows=[]
for f in [0.5,1,2,3,4,6,8,10]:
    cm=costs.scaled('xauusd',f)
    for tf,d,hold in [('H4',d4,1),('H1',d1,3)]:
        t=gold_session(d,hold_bars=hold)
        r=engine.simulate(d,t,'xauusd',cost_model=cm)
        st=metrics.stats(r.net)
        eff=(cm.spread/2+cm.slippage)/d.close.mean()*1e4*2+cm.commission_bps
        rows.append(dict(tf=tf, cost_x=f, spread=cm.spread, rt_cost_bps=round(eff,2),
                         cagr=100*st['cagr'], sharpe=st['sharpe'], maxdd=100*st['maxdd']))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n"+"="*112); print("C) YEAR-BY-YEAR (H4, 13.6y, net of costs)"); print("="*112)
t=gold_session(d4,hold_bars=1); r=engine.simulate(d4,t,'xauusd')
print(metrics.yearly(r.net).round(2).to_string())

print("\n"+"="*112); print("D) OUT-OF-SAMPLE SPLIT (H4): the effect was discovered on the whole sample,")
print("   so here is every 3-year block separately — no block should be carrying it alone.")
print("="*112)
rows=[]
for a,b in [(2013,2015),(2016,2018),(2019,2021),(2022,2023),(2024,2026)]:
    n=r.net[(r.net.index.year>=a)&(r.net.index.year<=b)]
    rows.append(dict(block=f"{a}-{b}", **{k:v for k,v in metrics.stats(n).items() if k in ('cagr','vol','sharpe','maxdd')}))
print(metrics.fmt(rows).to_string(index=False))
