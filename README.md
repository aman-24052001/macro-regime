# Adaptive Macro Regime Detection & Dynamic Factor Rotation

A production-quality quantitative research project that uses a **3-state Gaussian Hidden Markov Model** trained on 12 FRED macroeconomic series to dynamically rotate a multi-asset ETF portfolio (SPY / TLT / GLD / LQD / HYG / BIL) based on inferred economic regime probabilities.

**OOS period**: 2006-02 → 2026-02 (241 months) | **No look-ahead bias** (walk-forward expanding window + ALFRED vintage data)

---

## Key Results

| Strategy | CAGR | Vol | Max DD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|
| **Regime Strategy** | **8.45%** | **9.13%** | **-21.0%** | **0.937** | **1.357** | **0.402** |
| SPY Buy & Hold | 10.56% | 14.70% | -50.8% | 0.760 | 1.050 | 0.208 |
| 60/40 SPY/TLT | 8.56% | 9.81% | -28.5% | 0.890 | 1.115 | 0.300 |
| Equal Weight | 6.32% | 7.09% | -16.9% | 0.902 | 1.298 | 0.375 |

**Crisis performance** (vs SPY):
- Global Financial Crisis 2008–09: **+3.4%** (SPY: −46%)
- COVID Crash 2020: **−4.4%** (SPY: −9.2%)
- 2022 Rate Shock: **−16.3%** (SPY: −18.2%)

**Statistical robustness** (block bootstrap, n=1,000, block=12m):
Sharpe 95% CI [0.32, 1.17] · p(Sharpe > SPY) = 0.43

---

## Architecture

```
FRED API (12 series, 1960–)         yfinance (6 ETFs + 2 stocks)
        │                                      │
        ▼                                      ▼
  [Obtain]  fred_loader.py            market_loader.py
        │
        ▼
  [Scrub]   preprocessor.py → BME resample, forward-fill, rolling Z-score (36m)
            feature_engineer.py → 34 features (YoY, momentum, FSI composite)
        │
        ▼
  [Explore] eda_stats.py → ADF/KPSS stationarity, Spearman correlation, BIC elbow
            eda_plots.py → 4 EDA figures
        │
        ▼
  [Model]   hmm_selector.py → BIC over K=2..5 (50 restarts)
            hmm_trainer.py → Walk-forward expanding window (242 OOS months, 10 restarts/step)
                             → 7-feature input: yield curve, credit spread, unemployment,
                               CPI YoY, IP YoY, financial stress index, fed funds
            garch_model.py → GARCH(1,1) volatility overlay
        │
        ▼
  [Interpret] allocator.py → prob-weighted blend of 3 regime portfolios
              rebalancer.py → 2% threshold band + 7bps round-trip TC
              backtest.py + metrics.py → full performance attribution
              visualize.py → 10 output figures + tearsheet
        │
        ▼
  [Robustness] 108-combo grid: K × train_years × TC × threshold
  [Bootstrap]  Stationary block bootstrap (Politis & Romano 1994)
  [Overlay]    Stock-level 5-signal composite (NVDA, WDC case studies)
```

---

## Regime Portfolio Design

| Asset | Expansion | Stagnation | Contraction |
|---|---|---|---|
| SPY | 60% | 30% | 10% |
| TLT | 10% | 25% | 40% |
| GLD | 5% | 15% | 25% |
| LQD | 15% | 15% | 5% |
| HYG | 10% | 5% | 0% |
| BIL | 0% | 10% | 20% |

Weights at each month: `w(t) = P(expansion)·w_exp + P(stagnation)·w_stag + P(contraction)·w_cont`

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/macro-regime.git
cd macro-regime
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure FRED API key

Get a free API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

```bash
cp config/config.template.yaml config/config.yaml
# Edit config/config.yaml and replace ${FRED_API_KEY} with your key
```

Or use an environment variable:
```bash
export FRED_API_KEY="your_key_here"   # macOS/Linux
set FRED_API_KEY=your_key_here        # Windows
```

### 3. Run the full pipeline

```bash
python main.py --phase obtain        # Download FRED + market data (~2 min)
python main.py --phase scrub         # Feature engineering
python main.py --phase explore       # EDA + BIC elbow
python main.py --phase model         # Walk-forward HMM (~15 min)
python main.py --phase interpret     # Backtest + metrics + figures
python main.py --phase robustness    # 108-combo sensitivity grid (~20 min)
python main.py --phase bootstrap     # Block bootstrap CI (1,000 sims)
python main.py --phase overlay       # Stock overlay (NVDA, WDC)
```

Or run all at once:
```bash
python main.py --phase obtain scrub explore model interpret robustness bootstrap overlay
```

---

## Project Structure

```
macro_regime/
├── config/
│   ├── config.template.yaml     # Copy → config.yaml and add API key
│   └── config.yaml              # gitignored — contains API key
├── src/
│   ├── obtain/                  # FRED + yfinance data loaders
│   ├── scrub/                   # Preprocessing + feature engineering
│   ├── explore/                 # EDA statistics + plots
│   ├── model/                   # HMM selector, trainer, GARCH, bootstrap, robustness
│   ├── portfolio/               # Allocator + threshold rebalancer
│   └── interpret/               # Backtest engine, metrics, visualizations
├── scripts/
│   ├── inject_robustness.py     # Populate dashboard robustness section
│   └── inject_bootstrap.py      # Populate dashboard bootstrap section
├── notebooks/
│   ├── 01_obtain.ipynb          # OSEMN phase 1 walkthrough
│   ├── 02_scrub.ipynb
│   ├── 03_explore.ipynb
│   ├── 04_model.ipynb
│   └── 05_interpret.ipynb
├── tests/
│   └── test_pipeline.py         # 17 unit tests across 5 test classes
├── outputs/
│   ├── dashboard.html           # Full interactive dashboard (Chart.js)
│   ├── presentation.html        # Clean single-page interview deck
│   ├── figures/                 # 14 PNG charts (01–14)
│   └── reports/
│       ├── bootstrap_results.csv
│       ├── robustness_grid.csv
│       └── portfolio_returns.csv
├── status/
│   ├── PROJECT_JOURNEY.md       # Full development diary with bugs + fixes
│   └── interview_prep.md        # STAR framework, 25 Q&As, data section
├── main.py                      # CLI entry point
├── requirements.txt
└── requirements.lock
```

---

## Key Design Decisions & Anti-Overfitting Measures

| Decision | Rationale |
|---|---|
| Walk-forward expanding window | Never trains on future data; mimics live deployment |
| ALFRED vintage data (GDP, CPI) | Uses values as published on each date — eliminates revision bias |
| 7bps round-trip transaction cost | 5bps commission + 2bps slippage — realistic for ETF execution |
| 2% threshold rebalancing band | Avoids excessive turnover on small drift |
| `weights.shift(1)` before allocation | Ensures regime signal from month T uses weights from T+1 only |
| 36-month rolling Z-score | Normalizes features without using future data in scaling |
| Force K=3 (override BIC K=5) | Economic prior: Expansion/Stagnation/Contraction maps to real regimes |
| 50 restarts (BIC) / 10 (walk-fwd) | Escapes HMM local optima; verified n_fits sensitivity |
| Block bootstrap CI (block=12m) | Preserves annual seasonality; tests p(Sharpe > SPY benchmark) |

---

## Tech Stack

```
hmmlearn==0.3.2     Gaussian HMM
arch==6.2.1         GARCH(1,1) volatility
fredapi==0.5.1      FRED + ALFRED vintage data
yfinance>=0.2.35    ETF / stock price data
quantstats>=0.2.32  Full tearsheet generation
statsmodels==0.14.0 ADF / KPSS stationarity tests
scipy>=1.11         Scientific computing
matplotlib, seaborn Static charts
plotly>=5.17        Interactive charts
```

---

## Output Previews

The full interactive dashboard is at [`outputs/dashboard.html`](outputs/dashboard.html) — open in any browser.

The interview-ready single-page deck is at [`outputs/presentation.html`](outputs/presentation.html).

Selected figures:

| Figure | Description |
|---|---|
| `01_feature_timeseries.png` | All 7 HMM features over full history |
| `04_hmm_bic_elbow.png` | BIC elbow across K=2..5 |
| `05_regime_timeline.png` | Regime state timeline vs NBER recessions |
| `07_equity_and_drawdown.png` | Equity curve + drawdown vs benchmarks |
| `10_performance_comparison.png` | Side-by-side CAGR / Sharpe / Max DD bar chart |
| `13_robustness_grid.png` | 108-combo heatmap sensitivity analysis |
| `14_bootstrap_ci.png` | Bootstrap distributions: Sharpe, CAGR, Max DD |

---

## Limitations & Honest Caveats

- Bootstrap p-value (0.43) means the Sharpe advantage over SPY is not statistically significant at 95% — the strategy's edge is in **risk reduction** (half the drawdown, higher Sortino/Calmar), not raw return
- Robustness grid used n_fits=3 (vs production n_fits=10) for speed — absolute Sharpes in grid (0.55–0.77) are comparably lower but the relative parameter sensitivity is valid
- HYG and LQD only available from 2002 and 2003 respectively — early OOS months may have residual initialization effects
- GARCH volatility overlay uses realized vol, not forward-looking — still subject to vol regime breaks

---

## Author

**Aman** — Quantitative / Credit Analyst candidate

> Development journey, all bugs encountered, and fixes are documented in [`status/PROJECT_JOURNEY.md`](status/PROJECT_JOURNEY.md)
