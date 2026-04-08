# 🏎️ F1 Race Strategy AI

> An AI-powered Formula 1 race strategy dashboard built with Python,
> scikit-learn and Streamlit — using real F1 telemetry data from FastF1.

🔗 **[Live App](https://f1-strategy-ai-6hftkxuxmxzp2dogkyn5kf.streamlit.app/)**

---

![F1 Strategy AI Dashboard](assets/screenshot2.png)

---

## What It Does

This project uses machine learning to predict and optimise Formula 1
race strategies. It was built entirely from scratch using free tools
and publicly available F1 telemetry data across 46 Grand Prix events
from the 2022, 2023 and 2024 seasons.

The dashboard has six core features:

- **AI Optimal Strategy Finder** — exhaustively searches every valid
  1-stop and 2-stop strategy for the selected circuit and returns the
  fastest predicted combination
- **Custom Strategy Simulator** — input your own strategy and receive
  a predicted race time with a lap-by-lap breakdown
- **AI vs Your Strategy** — compare your strategy head-to-head against
  the AI's recommendation
- **Real vs Predicted Validation** — compares the AI's recommendation
  against what the 2024 race winner actually executed, validating the
  model against ground truth
- **Undercut / Overcut Analyser** — mid-race decision tool based on
  current lap, tyre age and gap to the car ahead
- **Optimal Fuel Load Calculator** — quantifies the time cost of
  carrying excess fuel at each circuit

A dynamic Safety Car probability banner updates for every circuit,
blending historical SC frequency data with track characteristic scores.

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Training data | 13,845 laps across 46 races |
| Seasons | 2022, 2023, 2024 |
| Circuits | 24 |
| R² Score | 0.991 |
| Mean Absolute Error | ±1.2s per lap |

---

## Screenshots

![Optimal Strategy](assets/screenshot1.png)
![Strategy Comparison](assets/screenshot3.png)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| FastF1 | Official F1 telemetry data |
| scikit-learn | Random Forest model |
| Streamlit | Web dashboard |
| Plotly | Interactive charts |
| pandas / numpy | Data processing |
| GitHub | Version control |
| Streamlit Cloud | Free deployment |

> All tools are free and open source. No paid APIs or cloud compute used.

---

## How to Run Locally

```bash
git clone https://github.com/vihaanishb-sudo/f1-strategy-ai.git
cd f1-strategy-ai
pip install -r requirements.txt
streamlit run App/dashboard.py
```

---

## Project Structure

f1-strategy-ai/
├── App/
│   └── dashboard.py          # Streamlit dashboard
├── Data/
│   ├── f1_laps_combined.csv  # Cleaned lap data
│   ├── circuit_strategies.json
│   ├── degradation_rates.json
│   ├── sc_probability.json
│   ├── winner_strategies.json
│   └── fuel_loads.json
├── Model/
│   ├── f1_strategy_model.pkl
│   └── feature_columns.pkl
├── requirements.txt
└── README.md

---

## About

Built by Vihaan — IBDP Year 1 student based in Cape Town, South Africa.
Developed as an independent project combining interests in Formula 1,
machine learning and software engineering.
