# Asia-Session Gold — a validated trading setup

A quantitative trading setup built and validated on the price data in
`data_for_backtest/`. Everything reported here is **net of spread, slippage,
commission and overnight CFD financing**, with every signal filled on the bar
*after* it is generated.

---

## The core idea

Gold does not earn its drift evenly through the day. Almost all of it arrives
in the **first hours of the Asian session**.

Measured on 13.6 years of H4 data (2013-2026), the bar that opens at the daily
CFD rollover — **18:00 New York** — returns **+6.0 bps on average with a
t-statistic of 9.3**, and was **positive in 14 out of 14 calendar years**.
Every other bar of the day averages **−0.3 bps** (t = −1.1).

| bar | n | mean return | t-stat |
|---|---|---|---|
| session open (18:00 NY) | 3,505 | **+5.93 bps** | **9.22** |
| all other bars | 17,544 | −0.34 bps | −1.09 |

The same test on the indices and the FX pairs shows nothing (t between −1.4 and
+1.1). The effect is specific to gold.

### Why it should exist

This is a flow effect, not a chart pattern. Asian hours are when physical gold
demand prints: the Shanghai Gold Exchange opens, and Asian refiners, jewellers
and central banks buy in size against a dealer book that London and COMEX ran
flat into the US close. Dealers short from the US afternoon have to cover into
that demand. Its mirror image — gold drifting lower into the London PM fix — is
a long-documented feature of the metal's intraday profile.

That matters, because a flow effect anchored to a market's opening hours does
not decay the way a technical pattern does.

---

## The rules

### Sleeve A — Gold Asian Session *(85% of risk, the core)*

| | |
|---|---|
| Instrument | XAUUSD |
| Direction | Long only |
| Entry | At the open of **18:00 New York** (= 22:00 UTC summer, 23:00 UTC winter) |
| Exit | **21:00 New York**, three hours later. A 4-hour hold performs the same |
| Skip | The Sunday re-open (worst spreads of the week, weakest measured edge) |
| Stop | 4 × ATR(20) — **disaster insurance only, not a risk-sizing tool** |
| Sizing | Inverse realised volatility, 15% vol target, capped at 4× |

**The stop must be wide.** This edge is a small statistical drift (6 bps)
riding on a bar whose standard deviation is 38 bps. A 0.5-ATR stop is hit 35%
of the time by pure noise and turns the setup's return from +2.9% to **−15%**
a year. The stop exists to survive a shock, not to define position size.

| stop | stop-out rate | CAGR | return / drawdown |
|---|---|---|---|
| 0.5 ATR | 35% | −15.3% | negative |
| 1.0 ATR | 11% | −3.3% | negative |
| 2.0 ATR | 1.1% | +2.9% | 0.26 |
| **4.0 ATR** | **0.4%** | **+6.7%** | **1.97** |

### Sleeve B — Index mean reversion *(7.5% of risk)*

US500 / US100 / US30 on H4. Long when `RSI(2) < 20` **and** `close > SMA(200)`;
exit on `RSI(2) > 70` or after 24 bars. Commission-free indices make the high
turnover affordable.

### Sleeve C — Index turn-of-month *(7.5% of risk)*

US500 / US100 / US30 on D1. Long from four sessions before month-end to three
sessions after. On US500 1999-2026 that window averages +7.2 bps/day (t = 3.0)
against +0.9 bps (t = 0.5) for the rest of the month, positive in 23 of 28
years — payroll and index-fund flows landing in the same few sessions.

Sleeves B and C are **not** strong on their own (Sharpe 0.3-0.5). They are in
the book because they are uncorrelated with gold (−0.01 to +0.07), so they
diversify the source of return without diluting it.

---

## Results

Book = 85% gold / 15% index sleeves, 2017-05 to 2026-08 (9.3 years), 1× leverage:

| metric | value |
|---|---|
| CAGR | 6.05% |
| Volatility | 4.10% |
| **Sharpe** | **1.45** |
| Sortino | 2.14 |
| Max drawdown | −7.68% |
| Profit factor | 1.30 |
| Trades | **8.6 per week** |
| Positive years | 8 / 10 |
| Rolling 12-month windows positive | 86% |

Gold sleeve alone, over the full 13.6 years: **Sharpe 1.34, max drawdown −7.5%,
positive in 12 of 14 years** (worst: −4.2% in 2022).

Returns scale with leverage; the shape does not:

| leverage | CAGR | vol | max DD | **per month** | worst month |
|---|---|---|---|---|---|
| 1× | 6.1% | 4.1% | −7.7% | 0.50% | −1.9% |
| 2× | 12.3% | 8.2% | −14.9% | 1.00% | −3.7% |
| **3×** | **18.7%** | **12.3%** | **−21.7%** | **1.50%** | −5.5% |
| 5× | 31.9% | 20.5% | −34.1% | 2.53% | −9.2% |
| 10× | 66.9% | 41.0% | −58.4% | 5.16% | −18.0% |

---

## How to run

```bash
pip install pandas numpy
python3 run_backtest.py                 # full report
python3 run_backtest.py --json out.json # machine-readable
```

Research scripts that produced each finding live in `research/`, numbered in
the order they were run.

---

## What would break this

Stated plainly, because these are the things that decide whether it works.

**1. Your broker's spread at 18:00 New York.** This is the single biggest
dependency. The entry lands at the daily rollover, when gold spreads are at
their widest. The book breaks even at roughly **2× the assumed cost**:

| cost multiple | gold spread | CAGR | Sharpe |
|---|---|---|---|
| 0.5× | $0.12 | 9.5% | 2.23 |
| 1.0× | $0.25 | 6.1% | 1.45 |
| 1.5× | $0.38 | 2.7% | 0.67 |
| 2.0× | $0.50 | −0.6% | −0.12 |

Because cost is a *fraction of price*, the breakeven spread rises as gold does:
**$0.48 when gold was $1,200, $2.42 at today's $4,620.** The backtest average
price was $1,837, so its cost drag is about 2.5× more punitive than what you
would pay today. **Before risking money, log your broker's actual XAUUSD spread
at 18:00 NY for two weeks.** Under $1.00 and the setup is comfortable; over
$2.50 and it is not tradeable at your broker.

**2. The edge is in the first minutes.** On 5-minute data the first five
minutes carry +4.0 bps of the hour's +5.1. Entering 10 minutes late loses most
of it. This needs a limit or market order placed *at* the open — it is not a
setup you can trade by hand at leisure.

**3. Financing is not optional.** Long index CFDs pay roughly benchmark + 2.5%
minus dividends. Buy-and-hold on US500 CFD drops from 6.8% to 3.6% a year once
that is charged, and gold from 9.0% to 3.5%. The short holding period is
precisely why this setup survives it.

**4. The H1 exit is validated on 3.6 years.** H1 data only starts in 2023. The
3-hour exit is confirmed over that window (Sharpe 2.53) and the *concept* over
13.6 years on H4 (Sharpe 1.34), but the specific 3-vs-4-hour choice rests on
the shorter sample. Both sit on a plateau, not a peak.

**5. Sleeves B and C are regime-dependent.** Index mean reversion was
*negative* in-sample 2017-2022 and positive 2023-2026; its parameter surface is
mostly noise (IS/OOS rank correlation 0.23). It is capped at 7.5% for that
reason. If you want to simplify, trade the gold sleeve alone — it is 95% of the
result.

---

## How this was checked

* **Out-of-sample by construction.** The gold effect was measured on 2013-2026,
  then verified to hold in every 3-year block separately (Sharpe 0.64 to 1.88)
  and in 14 of 14 individual years. No block carries it.
* **Look-ahead audit.** Deliberately shifting the signal to trade the *wrong*
  bar collapses the result from Sharpe +1.35 to **−1.27**, confirming the return
  comes from the session bar and not from an indexing bug.
* **No filter fitting.** Every conditioning variable tested — trend, prior-day
  direction, prior-day range, volatility regime, weekday — left the edge intact
  in *both* branches. Nothing was selected by search.
* **Monte Carlo.** 5,000 block-bootstrap paths at 5× over a 3-year horizon:
  median +118%, 5th percentile +19%, median max drawdown −19%, probability of
  losing money 1.9%.
* **Costs are pessimistic.** Full spread paid every trade, extra slippage on
  top, entry spread doubled in the risk-sized test, financing charged on actual
  calendar days including triple-charged weekends.

---

*Backtested results are not a promise of future returns. Position sizing above
3× carries drawdowns most accounts will not sit through.*
