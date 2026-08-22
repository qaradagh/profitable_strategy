import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import portfolio, metrics, data, engine, costs
from strategy.strategies.session import gold_session, session_open_mask
pd.set_option('display.width',260)

print("="*118); print("A) TAIL RISK — worst moves in the gold session bar, and what they cost at leverage")
print("="*118)
for tf,hold,lbl in [('240',1,'H4 4h window (2013+)'),('60',3,'H1 3h window (2023+)')]:
    d=data.load('xauusd',tf); m=session_open_mask(d.index)
    i=np.flatnonzero(m); i=i[i+hold<len(d)]
    entry=d['open'].to_numpy()[i]
    exitp=d['open'].to_numpy()[np.minimum(i+hold,len(d)-1)]
    ret=(exitp/entry-1)*100
    low=np.array([d['low'].to_numpy()[a:a+hold].min() for a in i])
    mae=(low/entry-1)*100
    print(f"\n{lbl}: n={len(ret)}")
    print(f"  window return  : mean {ret.mean():+.3f}%  std {ret.std():.3f}%  min {ret.min():+.2f}%  p1 {np.percentile(ret,1):+.2f}%  max {ret.max():+.2f}%")
    print(f"  worst adverse excursion inside the window: min {mae.min():.2f}%  p1 {np.percentile(mae,1):.2f}%")
    for L in [3,5,10]:
        print(f"    at {L}x notional on this trade, the worst historical window = {ret.min()*L:+.1f}% of account")

print("\n"+"="*118); print("B) COST SENSITIVITY of the whole book (gold .85 / idx .15)"); print("="*118)
import strategy.costs as C
orig=dict(C.COSTS)
rows=[]
for f in [0.5,1,1.5,2,3,4]:
    C.COSTS.update({k:C.scaled(k,f) for k in orig})
    _,comb=portfolio.build(portfolio.Config(w_gold=0.85,w_idx_mr=0.075,w_tom=0.075))
    t=comb['total']; t=t[t.index>='2017-05-01']
    st=metrics.stats(t)
    rows.append(dict(cost_multiple=f, cagr=100*st['cagr'], sharpe=st['sharpe'], maxdd=100*st['maxdd']))
C.COSTS.update(orig)
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n"+"="*118); print("C) LIVE SPEC on H1 (gold hold=3h, 2023-2026) vs the H4 long-history spec"); print("="*118)
rows=[]
for cfg,lbl in [(portfolio.Config(gold_tf='60',gold_hold=3,w_gold=0.85,w_idx_mr=0.075,w_tom=0.075),'H1 hold 3h'),
                (portfolio.Config(gold_tf='240',gold_hold=1,w_gold=0.85,w_idx_mr=0.075,w_tom=0.075),'H4 hold 4h')]:
    _,comb=portfolio.build(cfg); t=comb['total']; t=t[t.index>='2023-01-01']
    rows.append(metrics.summary_row(lbl+' (2023+)',t))
_,comb=portfolio.build(portfolio.Config(w_gold=0.85,w_idx_mr=0.075,w_tom=0.075))
rows.append(metrics.summary_row('H4 hold 4h (2017+)',comb['total'][comb['total'].index>='2017-05-01']))
print(metrics.fmt(rows)[['strategy','years','cagr','vol','sharpe','maxdd','calmar']].to_string(index=False))

print("\n"+"="*118); print("D) TRADE COUNT"); print("="*118)
sl,comb=portfolio.build(portfolio.Config(w_gold=0.85,w_idx_mr=0.075,w_tom=0.075))
tot=0; yrs=9.31
for k,v in sl.items():
    n=len(v.trades[v.trades.entry>='2017-05-01'])
    tot+=n; print(f"  {k:16s} {n:5d} trades  ({n/yrs:5.1f}/yr)")
print(f"  {'TOTAL':16s} {tot:5d} trades  ({tot/yrs:5.1f}/yr = {tot/yrs/52:.1f}/week)")

print("\n"+"="*118); print("E) LOOK-AHEAD AUDIT — shift every signal one extra bar; a real edge should survive")
print("="*118)
d=data.load('xauusd','240')
rows=[]
for lag in [0,1,2]:
    t=gold_session(d,hold_bars=1).shift(lag).fillna(0.0)
    r=engine.simulate(d,t,'xauusd'); st=metrics.stats(engine.to_daily(r.net))
    rows.append(dict(extra_lag_bars=lag, cagr=100*st['cagr'], sharpe=st['sharpe'], maxdd=100*st['maxdd']))
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("  lag=0 is the live rule. lag>=1 deliberately trades the WRONG bar; it should collapse,")
print("  which confirms the result comes from the session bar and not from a coding artefact.")
