"""What a small account actually earns from the gold session sleeve.

Sizing here is flat dollars per trade, not a percentage of equity, because
that is what an EA with a fixed risk input does: the lot size comes from
`risk_usd / (stop_atr * ATR)` and does not grow as the account grows.

The headline caveat this script exists to quantify: the entry lands on the
daily rollover, so the result is dominated by the broker's spread at that
moment, not by the size of the account.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data, costs
from strategy.strategies.session import session_open_mask
from strategy.indicators import atr
pd.set_option('display.width',240)
EQ0 = 3000.0

def entries(df):
    m=session_open_mask(df.index); s=np.zeros(len(df),bool); s[:-1]=m[1:]
    sun=np.asarray(df.index.dayofweek==6); ss=np.zeros(len(df),bool); ss[:-1]=sun[1:]
    return s & ~ss

def sim(tf, hold, risk_usd, cost_mult=1.0, entry_mult=2.0, stop_atr=4.0):
    """Flat risk_usd per trade. Position = risk_usd / (stop_atr * ATR)."""
    df=data.load('xauusd',tf); ent=entries(df)
    cm=costs.scaled('xauusd',cost_mult)
    o,h,lo=df.open.to_numpy(float),df.high.to_numpy(float),df.low.to_numpy(float)
    a=atr(df,20).to_numpy(float); idx=df.index; n=len(df)
    lr,_=costs.financing_rates('xauusd',idx,cm)
    hrs=np.zeros(n); hrs[:-1]=np.diff(idx.values).astype('timedelta64[s]').astype(float)/3600
    rows=[]
    for i in np.flatnonzero(ent):
        j=i+1
        if j>=n-1 or not np.isfinite(a[i]) or a[i]<=0: continue
        entry=o[j]; sd=stop_atr*a[i]; stop=entry-sd
        units=risk_usd/sd; notional=units*entry
        ec=units*(cm.spread/2*entry_mult+cm.slippage)+notional*cm.commission_bps/2/1e4
        end=min(j+hold,n-1); px,k=o[end],end; hit=False
        for k in range(j,end+1):
            if lo[k]<=stop: px=stop-cm.slippage; hit=True; break
        else: k=end
        xc=units*(cm.spread/2+cm.slippage)+units*px*cm.commission_bps/2/1e4
        carry=notional*lr[j]/360*(hrs[j:k].sum() if k>j else hrs[j])/24
        rows.append(dict(t=idx[k], pnl=units*(px-entry)-ec-xc-carry, notional=notional, hit=hit))
    b=pd.DataFrame(rows)
    m=b.set_index('t')['pnl'].resample('ME').sum()
    eq=EQ0+b.set_index('t')['pnl'].cumsum(); dd=eq/eq.cummax()-1
    return dict(mean=m.mean(), median=m.median(), worst=m.min(), best=m.max(),
                pos=(m>0).mean()*100, dd_usd=(dd*eq.cummax()).min(), dd_pct=dd.min()*100,
                lev=b.notional.mean()/EQ0, n=len(b), stops=b.hit.mean()*100,
                yrs=(b.t.iloc[-1]-b.t.iloc[0]).days/365.25, monthly=m)

print("="*100)
print(f"${EQ0:,.0f} ACCOUNT · FULLY MECHANICAL EA · flat $ risk per trade · 4xATR stop")
print("costs: full spread + slippage + commission every trade, ENTRY SPREAD DOUBLED at the rollover")
print("="*100)
rows=[]
for tf,hold,risk,lbl in [('60',3,150,'H1 spec (live rule), $150 risk — 2023-2026'),
                         ('240',1,150,'H4 spec (long validation), $150 risk — 2013-2026'),
                         ('240',1,271,'H4 spec, risk raised to match H1 exposure')]:
    r=sim(tf,hold,risk); r['spec']=lbl; rows.append(r)
t=pd.DataFrame(rows)[['spec','yrs','n','lev','mean','median','worst','best','pos','dd_usd','dd_pct','stops']]
t.columns=['spec','yrs','trades','notional_x','mean_$/mo','med_$/mo','worst_mo','best_mo','%mo+','maxDD_$','maxDD_%','%stopped']
print(t.round(1).to_string(index=False))

print("\n"+"="*100)
print("BROKER SPREAD AT 18:00 NY — the factor that decides the outcome")
print("(H4 spec, 13.6 years, exposure matched to the live H1 rule)")
print("="*100)
print(f"{'spread':>9} {'mean $/mo':>10} {'$/yr':>8} {'%mo+':>7} {'maxDD $':>10}")
for mult,sp in [(0.5,0.13),(1.0,0.25),(1.5,0.38),(2.0,0.50),(3.0,0.75),(5.0,1.25)]:
    r=sim('240',1,271,cost_mult=mult)
    print(f"{'$'+format(sp,'.2f'):>9} {r['mean']:>10.0f} {r['mean']*12:>8.0f} {r['pos']:>6.0f}% {r['dd_usd']:>10.0f}")

print("\n"+"="*100); print("RISK PER TRADE vs OUTCOME (H1 live rule, 2023-2026)"); print("="*100)
print(f"{'risk':>7} {'notional':>9} {'mean $/mo':>10} {'maxDD $':>9} {'maxDD %':>8}")
for rk in [50,100,150,250,400]:
    r=sim('60',3,rk)
    print(f"{'$'+str(rk):>7} {r['lev']:>8.1f}x {r['mean']:>10.0f} {r['dd_usd']:>9.0f} {r['dd_pct']:>7.1f}%")

r=sim('240',1,271)
print("\n"+"="*100); print("WORST STRETCHES on the 13.6-year sample (exposure-matched, $ on a $3,000 base)"); print("="*100)
m=r['monthly']; roll=m.rolling(12).sum()
print(f"  worst single month      ${m.min():>8,.0f}")
print(f"  worst 3-month stretch   ${m.rolling(3).sum().min():>8,.0f}")
print(f"  worst 12-month stretch  ${roll.min():>8,.0f}   ({(roll<0).sum()} of {roll.notna().sum()} rolling years were losers)")
print(f"  longest losing streak   {int((m<0).astype(int).groupby((m>=0).cumsum()).sum().max())} consecutive months")
