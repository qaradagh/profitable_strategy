"""Turn-of-month and calendar effects on indices (commission-free => cheap to trade)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
pd.set_option('display.width',250)
def tstat(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>2 and x.std()>0 else np.nan

print("="*115); print("A) TURN-OF-MONTH: daily log return by trading-day-of-month index")
print("   (tdom = 0 is the first trading day; negative counts back from month end)")
print("="*115)
for s in ['us500','us100','us30','xauusd']:
    d=data.load(s,'1D'); r=np.log(d.close/d.open)*1e4   # the daily bar's own body
    df=pd.DataFrame(dict(r=r)).dropna()
    ym=df.index.to_period('M')
    df['fwd_idx']=df.groupby(ym).cumcount()
    df['bwd_idx']=df.groupby(ym).cumcount(ascending=False)
    df['tdom']=np.where(df.fwd_idx<=6, df.fwd_idx, np.where(df.bwd_idx<=3, -1-df.bwd_idx, 99))
    g=df[df.tdom!=99].groupby('tdom')['r'].agg(n='size',bps='mean',t=tstat)
    print(f"\n--- {s.upper()} ({df.index[0].date()}..{df.index[-1].date()}) all-day mean={df.r.mean():.2f}bps ---")
    print(g.round(2).T.to_string())

print("\n"+"="*115); print("B) TURN-OF-MONTH WINDOW: hold long from tdom=-4..+3, by year (us500, us100)")
print("="*115)
for s in ['us500','us100']:
    d=data.load(s,'1D'); r=np.log(d.close/d.open)*1e4
    df=pd.DataFrame(dict(r=r)).dropna(); ym=df.index.to_period('M')
    df['f']=df.groupby(ym).cumcount(); df['b']=df.groupby(ym).cumcount(ascending=False)
    win=(df.f<=2)|(df.b<=3)
    inw=df.r[win]; outw=df.r[~win]
    print(f"{s}: in-window n={len(inw)} mean={inw.mean():.2f}bps t={tstat(inw):.2f} | "
          f"out n={len(outw)} mean={outw.mean():.2f}bps t={tstat(outw):.2f}")
    yr=df.assign(w=win,y=df.index.year).groupby(['y','w'])['r'].mean().unstack()
    yr.columns=['out','in']
    print("  positive years in-window:", int((yr['in']>0).sum()), "/", len(yr))

print("\n"+"="*115); print("C) SAME on H4 for indices: is the turn-of-month concentrated in a session?")
print("="*115)
for s in ['us500']:
    d=data.load(s,'240'); b=np.log(d.close/d.open)*1e4
    ny=d.index.tz_convert('America/New_York')
    df=pd.DataFrame(dict(r=b.values, day=d.index.normalize()), index=d.index).dropna()
    dd=pd.Series(df.day.unique()).sort_values()
    ym=pd.DatetimeIndex(dd).to_period('M')
    f=pd.Series(range(len(dd)),index=dd).groupby(ym.values).rank(method='first')-1
    bwd=pd.Series(range(len(dd)),index=dd).groupby(ym.values).rank(method='first',ascending=False)-1
    win_days=set(dd[(f<=2)|(bwd<=3)])
    df['win']=df.day.isin(win_days)
    df['nyh']=ny.hour
    print(df.groupby(['win','nyh'])['r'].agg(n='size',bps='mean',t=tstat).round(2).to_string())
