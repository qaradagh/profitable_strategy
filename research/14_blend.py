import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data, engine, metrics
from strategy.strategies.session import gold_session
from strategy.strategies.seasonal import turn_of_month
from strategy.strategies import index_mean_reversion
pd.set_option('display.width',250)

def daily(x): return x.groupby(x.index.normalize()).sum()

sleeves={}
d4=data.load('xauusd','240')
sleeves['gold_session_H4']=daily(engine.simulate(d4,gold_session(d4,hold_bars=1),'xauusd').net)
d1=data.load('xauusd','60')
sleeves['gold_session_H1']=daily(engine.simulate(d1,gold_session(d1,hold_bars=3),'xauusd').net)
for s in ['us500','us100','us30']:
    dd=data.load(s,'240')
    t=index_mean_reversion(dd,rsi_len=2,entry=20,exit_rsi=70,exit_ma=10**9,max_hold=24,trend_len=200,stop=0.05)
    sleeves[f'idxMR_{s}']=daily(engine.simulate(dd,t,s).net)
    dv=data.load(s,'1D')
    sleeves[f'TOM_{s}']=daily(engine.simulate(dv,turn_of_month(dv),s).net)

print("="*115); print("INDIVIDUAL SLEEVES (net of costs, 1x notional)"); print("="*115)
rows=[metrics.summary_row(k,v) for k,v in sleeves.items()]
print(metrics.fmt(rows)[['strategy','start','end','years','cagr','vol','sharpe','maxdd','calmar','exposure']].to_string(index=False))

print("\n"+"="*115); print("CORRELATION of daily returns (common window 2017+)"); print("="*115)
M=pd.DataFrame(sleeves).fillna(0.0)
M=M[M.index>='2017-05-01']
print(M.corr().round(2).to_string())

print("\n"+"="*115); print("VOL-SCALING the gold session sleeve: does inverse-vol sizing help?"); print("="*115)
from strategy.indicators import realized_vol
rows=[]
base=gold_session(d4,hold_bars=1)
rows.append(dict(**metrics.summary_row('gold session, flat size', daily(engine.simulate(d4,base,'xauusd').net))))
for n in [20,60,120]:
    for tgt in [0.10,0.15]:
        rv=realized_vol(d4.close,n,periods_per_year=6*252).shift(1)
        scale=(tgt/rv).clip(upper=4.0).fillna(0.0)
        t=base*scale
        rows.append(dict(**metrics.summary_row(f'gold session, invvol n={n} tgt={tgt:.0%}', daily(engine.simulate(d4,t,'xauusd').net))))
print(metrics.fmt(rows)[['strategy','years','cagr','vol','sharpe','maxdd','calmar']].to_string(index=False))
