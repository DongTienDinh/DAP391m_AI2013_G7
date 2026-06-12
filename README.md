# COMPASS-XAI: Composite Market Expansion Scoring with Aligned SHAP Explanations

An end-to-end machine learning and strategic decision-support platform for e-commerce geographic expansion in Brazil. Forecasts weekly demand, ranks states by **Expansion Priority Score (EPS)** using Shannon Entropy optimization, and explains decisions via SHAP-aligned Gemini narratives.

---

## Overview

1. **Stable Forecasting** — State-level weekly revenue predictions using 9 regression models with walk-forward CV.
2. **Strategic Ranking** — The **EPS** composite index (Demand, Growth, Penetration, Momentum, Logistics Risk) optimized via SLSQP Entropy Maximization.
3. **Explainable AI** — Strategic narratives via Gemini LLM or rule-based fallback, aligned with SHAP feature attributions.
4. **Interactive Dashboard** — Streamlit "Command Center" with real-time what-if sliders, choropleth map, radar charts, and chatbot assistant.

---

## Datasets & Sources

| Source | Description |
|--------|-------------|
| [Olist E-Commerce (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | ~100k delivered orders, Brazil, 2016–2018 |
| [IBGE Population](https://basedosdados.org/dataset/d30222ad-7a5c-4778-a1ec-f0785371d1ca?table=b99f0017-e587-477e-8cfb-05fb5d1005b8) | State-level populations |
| [IBGE GDP](https://basedosdados.org/dataset/fcfcf025ca-8b19-4131-8e2d-5ddb12492347?table=93007431-7ce9-42ee-8740-8c2274d345ad) | State-level GDP |
| Brazil GeoJSON | State boundaries (auto-downloaded) |

---

## Project Structure

```
src/
├── analysis/               # Scoring, optimization, XAI, ranking comparison
│   ├── engine.py           # EPS component formulas (PD, GP, PG)
│   ├── optimizer.py        # SLSQP Shannon Entropy maximization
│   ├── scoring.py          # Scoring service orchestrator
│   ├── shap.py             # SHAP explainer
│   ├── xai.py              # Narrative synthesis (LLM + rule-based)
│   ├── ranking_comparison.py  # EPS vs 4 baselines comparison
│   └── providers/          # LLM providers (Gemini)
├── core/                   # Config, constants, exceptions, logging
├── data/                   # Ingestion, cleaning, schema validation
├── features/               # State-week panel builder, transformers
├── models/                 # Model factory, evaluator, training
├── utils/                  # Chatbot, config loader, math utils
├── scripts/                # Executable pipeline entrypoints
│   ├── run_cleaning.py     # Data cleaning
│   ├── run_features.py     # Feature engineering
│   ├── run_training.py     # Model training & evaluation
│   ├── run_scoring.py      # EPS scoring & ranking
│   ├── run_xai.py          # XAI narrative generation
│   ├── run_figures.py      # Static figure generation (maps, charts)
│   ├── run_ranking_comparison.py  # EPS vs baseline analysis
│   └── download_geojson.py # Brazil GeoJSON download
├── orchestrator.py         # E2E pipeline orchestrator
└── pipeline.py             # Interactive CLI pipeline (10-step)
app/                        # Streamlit dashboard
configs/                    # YAML configs (paths, training, inference)
tests/                      # Unit & integration tests
```

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/DongTienDinh/DAP391m_AI2013_G7
cd DAP391m_AI2013_G7
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Setup

Place IBGE CSV files in `data/external/`:
- `br_ibge_populacao_uf.csv`
- `br_ibge_pib_uf.csv`

### 3. Run Pipeline

**Interactive mode (recommended):**
```bash
python src/pipeline.py
```
Prompts for each phase and shows results after each step.

**Automated mode:**
```bash
python src/pipeline.py --all
```

**Or run individual scripts:**
```bash
python src/scripts/download_geojson.py     # Brazil map boundaries
python src/scripts/run_cleaning.py          # Download + clean Olist data
python src/scripts/run_features.py          # Build feature panel
python src/scripts/run_training.py          # Train 9 models
python src/scripts/run_scoring.py           # EPS rankings
python src/scripts/run_figures.py           # Generate static figures
python src/scripts/run_xai.py               # XAI narratives (needs GEMINI_API_KEY)
```

### 4. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Pipeline Steps (DAP391m 10-Step Framework)

| Step | Phase | Script | Description |
|------|-------|--------|-------------|
| 0 | Setup | `download_geojson.py` | Brazil state boundaries |
| 1–5 | Data | `run_cleaning.py` | Kaggle download, schema validation, cleaning |
| 6 | Features | `run_features.py` | 35+ features: lags, rolling, IBGE, seasonality |
| 7–8 | Training | `run_training.py` | 9 models, 5-fold walk-forward CV |
| 9 | Scoring | `run_scoring.py` | SLSQP entropy optimization, EPS ranking |
| 9b | Comparison | `run_ranking_comparison.py` | EPS vs 4 progressive baselines |
| 9c | Figures | `run_figures.py` | Choropleth maps, radar charts, correlation heatmap |
| 10 | XAI | `run_xai.py` | Rule-based + Gemini narratives, SHAP alignment |

---

## Methodology: Expansion Priority Score (EPS)

$$EPS = (\sum w_i \cdot \text{Component}_i) \times (1 - \gamma \cdot \text{LC}_{\text{norm}})$$

| Component | Definition |
|-----------|-----------|
| **PD** (Predicted Demand) | `0.5 * ln(1 + predicted_rev) + 0.5 * ln(1 + actual_4w)` |
| **GP** (Growth Potential) | Cycle-over-cycle revenue growth, clipped [-1, 1] |
| **PG** (Penetration Gap) | GDP-adjusted untapped market potential |
| **MMI** (Market Momentum) | Revenue efficiency per active seller |
| **LC** (Logistics Cost) | Normalized freight cost (risk penalty) |

**Weight Optimization:** Maximizes Shannon Entropy via SLSQP:

$$\max_w -\sum p_j \log p_j, \quad \text{s.t. } \sum w_i = 1, \quad l_i \leq w_i \leq u_i$$

---

## Model Benchmark Results

| Rank | Model | RMSE | MAE | Skill Score |
|:----:|-------|-----:|----:|:-----------:|
| #1 | CatBoost | 50.11 | 22.28 | +0.999 |
| #2 | LightGBM | 51.94 | 23.48 | +0.999 |
| #3 | Gradient Boosting | 52.33 | 23.23 | +0.999 |
| #4 | Random Forest | 53.37 | 23.93 | +0.999 |
| #5 | XGBoost | 55.23 | 24.73 | +0.999 |
| 6 | ElasticNet | 3,809,672.58 | 466,000.43 | +0.889 |
| 7 | Ridge | 9,942,046.88 | 1,201,996.36 | +0.711 |
| 8 | Huber | 262,323,614.33 | 23,254,81.62 | -6.628 |

*Metrics on original scale after np.expm1 inverse transform. Walk-forward CV (5-fold).*

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/eps/eps_results.csv` | Full state ranking with component scores |
| `outputs/eps/w_star.json` | Optimal component weights |
| `outputs/eps/eps_xai_report.json` | State-by-state XAI narratives |
| `reports/model_leaderboard.csv` | Model comparison metrics |
| `reports/ranking_comparison.csv` | EPS vs 4 baselines comparison |
| `reports/figures/fig*.png` | Choropleth, radar, correlation, sensitivity figures |

---

## Additional Features

- **AI Chatbot** — Ask questions about data, source code, and reports via Gemini (click *Ask AI* in dashboard header)
- **Ranking Comparison** — Compare EPS against revenue-only, forecast-only, OPP-only, and logistics-adjusted baselines
- **Sensitivity Analysis** — Monte Carlo robustness, OAT weight sweeps, gamma penalty sweeps
- **Interactive Map** — Click any state on the choropleth to deep-dive

---

## License

MIT License. See `LICENSE` for details.
