import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from strategy import portfolio, metrics, data, engine
pd.set_option('display.width',260)
W='2017-05-01'

print("="*118); print("A) SAME-WINDOW COMPARISON (2017-05 .. 2026-08) — do the index sleeves earn their place?")
print("="*118)
variants={
 'gold only'            : portfolio.Config(use_index_sleeves=False, w_gold=1.0),
 'gold .85 / idx .15'   : portfolio.Config(w_gold=0.85, w_idx_mr=0.075, w_tom=0.075),
 'gold .70 / idx .30'   : portfolio.Config(w_gold=0.70, w_idx_mr=0.15,  w_tom=0.15),
 'gold .50 / idx .50'   : portfolio.Config(w_gold=0.50, w_idx_mr=0.25,  w_tom=0.25),
 'index sleeves only'   : portfolio.Config(w_gold=0.0,  w_idx_mr=0.5,   w_tom=0.5),
}
tots={}
rows=[]
for k,c in variants.items():
    _,comb=portfolio.build(c)
    t=comb['total']; t=t[t.index>=W]
    tots[k]=t
    rows.append(metrics.summary_row(k,t))
print(metrics.fmt(rows)[['strategy','years','cagr','vol','sharpe','sortino','maxdd','calmar','profit_factor']].to_string(index=False))

print("\n"+"="*118); print("B) WALK-FORWARD: re-fit nothing, but check each year out of sample (gold .85/idx .15)")
print("="*118)
base=tots['gold .85 / idx .15']
yr=metrics.yearly(base)
print(pd.DataFrame({'year':yr.index.year,'return_%':yr.values.round(2)}).to_string(index=False))

print("\n"+"="*118); print("C) ROLLING 12-MONTH SHARPE & RETURN (gold .85/idx .15)"); print("="*118)
r12=base.rolling(252)
s12=(r12.mean()/r12.std()*np.sqrt(252)).dropna()
ret12=((1+base).rolling(252).apply(np.prod,raw=True)-1).dropna()*100
print(f"rolling 12m Sharpe: min {s12.min():.2f}  p10 {s12.quantile(.1):.2f}  median {s12.median():.2f}  max {s12.max():.2f}")
print(f"rolling 12m return: min {ret12.min():.2f}%  p10 {ret12.quantile(.1):.2f}%  median {ret12.median():.2f}%  max {ret12.max():.2f}%")
print(f"share of rolling 12m windows positive: {(ret12>0).mean():.1%}")

print("\n"+"="*118); print("D) LEVERAGE FRONTIER — what you actually get at each risk level"); print("="*118)
fr=portfolio.frontier(base,[1,2,3,4,5,6,8,10,12])
fr2=fr.copy()
for c in ['cagr','vol','maxdd','monthly_mean','monthly_median','worst_month','pct_months_pos']:
    fr2[c]=(fr2[c]*100).round(2)
print(fr2.round(2).to_string(index=False))

print("\n"+"="*118); print("E) MONTE CARLO — block bootstrap of daily returns (5000 paths, 3y horizon, at 5x)")
print("="*118)
rng=np.random.default_rng(7)
x=(base*5).to_numpy(); n=len(x); block=20; horizon=756
finals=[]; dds=[]
for _ in range(5000):
    idxs=rng.integers(0,n-block,size=horizon//block+1)
    path=np.concatenate([x[i:i+block] for i in idxs])[:horizon]
    eq=np.cumprod(1+path)
    finals.append(eq[-1]-1); dds.append((eq/np.maximum.accumulate(eq)-1).min())
finals=np.array(finals); dds=np.array(dds)
print(f"3-year total return: p5 {np.percentile(finals,5)*100:7.1f}%  p25 {np.percentile(finals,25)*100:7.1f}%  "
      f"median {np.percentile(finals,50)*100:7.1f}%  p75 {np.percentile(finals,75)*100:7.1f}%  p95 {np.percentile(finals,95)*100:7.1f}%")
print(f"max drawdown       : p50 {np.percentile(dds,50)*100:7.1f}%  p75 {np.percentile(dds,75)*100:7.1f}%  "
      f"p95 {np.percentile(dds,95)*100:7.1f}%  worst {dds.min()*100:7.1f}%")
print(f"probability of losing money over 3 years: {(finals<0).mean():.1%}")
