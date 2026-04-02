# 🫀 Digital Twin Simulator — Obesity-Induced Liver Disease Risk

> ** Production-Grade · Bayesian Probabilistic AI**
An AI-powered digital twin that simulates how lifestyle changes like diet, exercise, and alcohol affect liver disease risk over time.

Users can explore how improving habits reduces risk and improves liver health.
A full-stack, production-quality digital twin that simulates liver disease risk driven by obesity and lifestyle factors, using a Probabilistic Machine Learning with Uncertainty Quantification (Bootstrap-based), inspired by Bayesian principles having confidence intervals and time-evolution simulation.

---

## 📌 Features

| Feature | Detail |
|---|---|
| **Dataset** | ILPD (Indian Liver Patient Dataset) — 416 real patients, expanded 6× synthetically |
| **BMI Engine** | Realistic height/weight generation; BMI categorisation (WHO) |
| **Bayesian Model** | Calibrated Gradient Boosting + 80 bootstrap models for CI |
| **Causal DAG** | BMI → Obesity → Liver Risk; Alcohol → Liver Damage; Diet/Exercise → Protection |
| **Confidence Intervals** | 90% CI from bootstrap ensemble on every prediction |
| **Time Evolution** | Month-by-month physiological simulation (BMI, enzymes, albumin) |
| **Scenario Comparison** | 4 lifestyle scenarios plotted simultaneously |
| **Streamlit App** | Dark-theme, interactive, slider-driven with Plotly charts |
| **Docker** | One-command deployment |

---

## 🏗 Project Structure

```
digital_twin_liver/
│
├── data/
│   ├── generate_ilpd.py        ← Synthetic ILPD generator (if CSV missing)
│   ├── ILPD.csv                ← Raw input (auto-generated if absent)
│   ├── Cleaned_data.csv        ← After preprocess.py
│   ├── Synthetic_data.csv      ← 6× synthetic expansion
│   ├── Combined_dataset.csv    ← Full training corpus
│   └── liver_model.joblib      ← Trained model + bootstrap ensemble
│
├── src/
│   ├── preprocess.py           ← Load, clean, encode, scale ILPD
│   ├── synthetic_data.py       ← Expand dataset, add lifestyle features
│   ├── bmi.py                  ← Height/weight generation, BMI calculation
│   ├── model.py                ← Bayesian probabilistic model + CI
│   └── simulation.py           ← Digital twin time-evolution engine
│
├── app/
│   └── streamlit_app.py        ← Interactive Streamlit dashboard
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A — Local Python

```bash
# 1. Clone / download
cd digital_twin_liver

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch app (auto-trains model on first run)
streamlit run app/streamlit_app.py
```

Open: `http://localhost:8501`

---

### Option B — Docker

```bash
# Build image (pre-trains model during build)
docker build -t liver-twin .

# Run
docker run -p 8501:8501 liver-twin

# OR with persistent data volume
docker run -p 8501:8501 -v $(pwd)/data:/app/data liver-twin
```

Open: `http://localhost:8501`

---

### Option C — Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set **Main file**: `app/streamlit_app.py`
4. Deploy

---

## 🧬 Model Architecture

```
Raw ILPD Data (416 patients)
        ↓
   preprocess.py
   • Rename columns
   • Encode gender (M=1, F=0)
   • Map target (1=disease, 2=healthy → 1/0)
   • Impute missing (median)
   • Clip 99th-pct outliers
        ↓
   synthetic_data.py  (×6 expansion)
   • Preserve correlations via Gaussian copula noise
   • Add lifestyle features (Beta distribution)
        ↓
   bmi.py
   • Generate realistic height/weight (Indian population norms)
   • Compute BMI; WHO categorisation
        ↓
   model.py  — Feature Engineering
   • Log-transform enzymes (right-skewed)
   • enzyme_score = mean(log_SGPT, log_SGOT, log_Bili, log_ALP)
   • obesity_risk  = clip((BMI - 18.5) / 21.5, 0, 1)
   • lifestyle_risk = alcohol×0.5 − diet×0.25 − exercise×0.25
        ↓
   CalibratedClassifierCV(GradientBoostingClassifier)
   • 200 estimators, lr=0.05, max_depth=4
   • Isotonic calibration (5-fold CV)
   • 80 bootstrap resamples → 90% CI
        ↓
   simulation.py  — Digital Twin
   • State S_t = {biomarkers, lifestyle}
   • S_{t+1} = physiological update rules
   • P(disease | S_t) from model at each step
```

---

## 📊 Streamlit App Tabs

| Tab | Content |
|---|---|
| **Risk Dashboard** | Gauge chart, 90% CI, risk radar, clinical markers table |
| **Time Evolution** | Risk + biomarker charts over simulation horizon |
| **Scenario Comparison** | 4 lifestyle trajectories overlaid, end-state table |
| **Dataset Explorer** | Distribution plots, correlation heatmap, downloadable CSV |

### Sidebar Inputs

- **Age** slider (18–85)
- **Weight** + **Height** → auto BMI
- **Liver panel**: Bilirubin, ALP, SGPT, SGOT, Proteins, Albumin, A/G Ratio
- **Lifestyle**: Exercise, Alcohol, Diet (0–1 scales)
- **Simulation months**: 6–36

---

## 🔬 Dataset Details

**ILPD — Indian Liver Patient Dataset**

| Column | Description |
|---|---|
| Age | Patient age |
| Gender | Male / Female |
| Total_Bilirubin | mg/dL |
| Direct_Bilirubin | mg/dL |
| Alkaline_Phosphotase | IU/L |
| Alamine_Aminotransferase | SGPT IU/L |
| Aspartate_Aminotransferase | SGOT IU/L |
| Total_Protiens | g/dL |
| Albumin | g/dL |
| Albumin_and_Globulin_Ratio | ratio |
| Dataset | 1=liver disease, 2=no liver disease |

**Synthetic Additions**: `exercise_level`, `alcohol_intake`, `diet_quality`, `height_m`, `weight_kg`, `bmi`, `bmi_category`

---

## ⚙️ Run Individual Pipeline Steps

```bash
# Generate ILPD CSV
python data/generate_ilpd.py

# Preprocess
python src/preprocess.py

# Synthetic data
python src/synthetic_data.py

# BMI
python src/bmi.py

# Train model
python src/model.py

# Simulation demo
python src/simulation.py
```

---

## 📈 Model Performance (on held-out 20%)

| Metric | Score |
|---|---|
| ROC-AUC | ~0.82–0.87 |
| Average Precision | ~0.88–0.92 |
| CI Coverage (90%) | ~90% empirical |

*(Varies slightly each run due to stochastic training)*

---

## 📸 Demo

### Dashboard Overview
![Dashboard](assets/Dashboard%20screenshot.png)

### Risk Simulation
![Dashboard](assets/dashboard%20screenshot%202.png)

### Model Insights
![Dashboard](assets/dashboard%20screenshot%203.png)

## 🤝 Contributing

Pull requests welcome. Key areas for extension:
- SHAP explainability integration
- PyMC3/Stan true Bayesian network
- Longitudinal real patient data ingestion
- REST API wrapper (FastAPI)

---

## 📄 License

MIT License — free for academic and commercial use.

---

*Built for IIT BHU Hackathon 2025 ·  Health Track*
