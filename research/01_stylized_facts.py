"""Baseline stylized facts: drift, vol, autocorrelation, session split."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data

pd.set_option('display.width', 220); pd.set_option('display.float_format', lambda x: f'{x:9.4f}')

print("="*100); print("A) DAILY BUY & HOLD CHARACTERISTICS (full D1 history)"); print("="*100)
rows=[]
for s in data.SYMBOLS:
    d = data.load(s,'1D')
    r = np.log(d.close).diff().dropna()
    for label, rr in [("full", r), ("2013+", r[r.index>='2013-01-01']), ("2023+", r[r.index>='2023-01-01'])]:
        rows.append(dict(sym=s, span=label, yrs=round((rr.index[-1]-rr.index[0]).days/365.25,1),
            ann_ret=rr.mean()*252, ann_vol=rr.std()*np.sqrt(252),
            sharpe=rr.mean()/rr.std()*np.sqrt(252), skew=rr.skew(),
            ac1=rr.autocorr(1), ac2=rr.autocorr(2), ac5=rr.autocorr(5)))
print(pd.DataFrame(rows).to_string(index=False))

print(); print("="*100); print("B) OVERNIGHT vs INTRADAY on indices (H1 data, 2023+)"); print("="*100)
# Define 'day' by the exchange cash session. CFD index bars run ~23:00 UTC -> 21:00 UTC.
for s in data.INDICES + ['xauusd']:
    h = data.load(s,'60')
    d = data.load(s,'1D')
    d = d[d.index >= h.index[0].normalize()]
    o, c = d.open, d.close
    intraday = np.log(c/o)                       # open -> close on the daily bar
    overnight = np.log(o/c.shift(1)).dropna()    # prev close -> open
    tot = np.log(c).diff().dropna()
    def st(x, nm):
        x=x.dropna()
        return dict(sym=s, leg=nm, n=len(x), ann=x.mean()*252, vol=x.std()*np.sqrt(252),
                    sharpe=x.mean()/x.std()*np.sqrt(252), t=x.mean()/x.std()*np.sqrt(len(x)),
                    hit=(x>0).mean())
    print(pd.DataFrame([st(overnight,'overnight'), st(intraday,'intraday'), st(tot,'total')]).to_string(index=False))

print(); print("="*100); print("C) HOUR-OF-DAY mean log return (bps), H1 2023+"); print("="*100)
tab={}
for s in data.SYMBOLS:
    h=data.load(s,'60'); r=np.log(h.close).diff()
    tab[s]=(r.groupby(r.index.hour).mean()*1e4)
print(pd.DataFrame(tab).round(2).to_string())

print(); print("="*100); print("D) DAY-OF-WEEK mean log return (bps), D1 2013+"); print("="*100)
tab={}
for s in data.SYMBOLS:
    d=data.load(s,'1D'); d=d[d.index>='2013-01-01']; r=np.log(d.close).diff()
    tab[s]=(r.groupby(r.index.dayofweek).mean()*1e4)
print(pd.DataFrame(tab).round(2).to_string())
