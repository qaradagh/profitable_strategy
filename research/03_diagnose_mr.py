import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data, engine, metrics
from strategy.strategies import index_mean_reversion
pd.set_option('display.width',220)

s='us500'; d=data.load(s,'1D')
t=index_mean_reversion(d)
res=engine.simulate(d,t,s)
f=res.frame
print("--- P&L decomposition, MR us500, full history ---")
yrs=(f.index[-1]-f.index[0]).days/365.25
print(f"gross ann : {f.gross.sum()/yrs*100:7.2f}%")
print(f"spread    : {-f.trade_cost.sum()/yrs*100:7.2f}%")
print(f"financing : {-f.carry.sum()/yrs*100:7.2f}%")
print(f"net  ann  : {f.net.sum()/yrs*100:7.2f}%")
print(f"trades    : {len(res.trades)}  ({len(res.trades)/yrs:.1f}/yr), avg bars {res.trades.bars.mean():.1f}")
print(f"avg trade : {res.trades.pnl.mean()*1e4:.1f} bps   win {100*(res.trades.pnl>0).mean():.1f}%")

print("\n--- by decade (gross bps per trade) ---")
tr=res.trades.copy(); tr['dec']=tr.entry.dt.year//5*5
print(tr.groupby('dec').agg(n=('pnl','size'), bps=('pnl',lambda x:x.mean()*1e4), win=('pnl',lambda x:(x>0).mean())).round(2).to_string())

print("\n--- WHERE DOES THE MOVE HAPPEN? (signal at close[t], RSI2<10 & >SMA200) ---")
from strategy.indicators import rsi
c=d.close; R=rsi(c,2); ma=c.rolling(200).mean()
sig=(c>ma)&(R<10)&ma.notna()
o,cl=d.open,d.close
legs={
 'overnight close[t]->open[t+1]': np.log(o.shift(-1)/cl),
 'day1 open[t+1]->close[t+1]'   : np.log(cl.shift(-1)/o.shift(-1)),
 'day2 open[t+1]->open[t+2]'    : np.log(o.shift(-2)/o.shift(-1)),
 'day1-3 open[t+1]->open[t+4]'  : np.log(o.shift(-4)/o.shift(-1)),
 'day1-5 open[t+1]->open[t+6]'  : np.log(o.shift(-6)/o.shift(-1)),
 'close[t]->close[t+5]'         : np.log(cl.shift(5).shift(-5)/cl) if False else np.log(cl.shift(-5)/cl),
}
rows=[]
for k,v in legs.items():
    x=v[sig].dropna()
    rows.append(dict(leg=k,n=len(x),bps=x.mean()*1e4,t=x.mean()/x.std()*np.sqrt(len(x)),hit=(x>0).mean()))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n--- SPREAD BURDEN over time (0.8 pts round-half on us500 price) ---")
for y in [2000,2005,2010,2015,2020,2026]:
    px=c[c.index.year==y].mean()
    print(f"  {y}: price {px:8.1f}  round-turn cost {2*0.8/px*1e4:5.2f} bps")
