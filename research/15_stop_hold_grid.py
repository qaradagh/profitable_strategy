import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import data
from strategy.risk_backtest import run_trades, RiskConfig, summarise
from strategy.strategies.session import gold_session
pd.set_option('display.width',260)

rows=[]
for tf,holds in [('240',[1,2,3]), ('60',[2,3,4,6,8])]:
    d=data.load(tf and 'xauusd',tf)
    ent=(gold_session(d,hold_bars=1).to_numpy()>0)
    for hold in holds:
        for stop in [1.5,2.0,3.0,4.0,6.0,99.0]:
            cfg=RiskConfig(risk_pct=0.01,stop_atr=stop,hold_bars=hold,entry_spread_mult=2.0)
            b,_=run_trades(d,ent,'xauusd',cfg)
            s=summarise(b)
            if not s: continue
            rows.append(dict(tf=tf,hold=hold,stop=stop, cagr=s['cagr'], mo=s['monthly_mean'],
                             dd=s['max_dd'], lev=s['avg_leverage'], stopr=s['stop_rate'],
                             win=s['win_rate'], avgR=s['avg_R'], pf=s['profit_factor'],
                             mo_pos=s['pct_months_positive'],
                             ret_dd=s['cagr']/abs(s['max_dd']) if s['max_dd'] else np.nan))
g=pd.DataFrame(rows)
print("=== risk_pct=1% per trade, entry spread x2 (pessimistic). 'lev' = avg notional/equity ===")
for tf in ['240','60']:
    print(f"\n--- {'H4 (2013-2026, 13.6y)' if tf=='240' else 'H1 (2023-2026, 3.6y)'} ---")
    sub=g[g.tf==tf]
    print(sub.round(4).to_string(index=False))
print("\n=== return/drawdown ratio surface (higher is better) ===")
for tf in ['240','60']:
    print(f"\n{tf}:"); print(g[g.tf==tf].pivot_table(index='hold',columns='stop',values='ret_dd').round(2).to_string())
g.to_csv('research/_stop_hold.csv',index=False)
