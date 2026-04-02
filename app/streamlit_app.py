"""
streamlit_app.py
────────────────
Digital Twin Simulator for Obesity-Induced Liver Disease Risk
IIT BHU Hackathon — Production Streamlit App
"""

import sys
import os
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
SRC  = ROOT / "src"
DATA = ROOT / "data"
sys.path.insert(0, str(SRC))

import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liver Digital Twin",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e8eaf6;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}

.metric-card {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1525 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
}

.metric-label {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7986cb;
    margin-top: 6px;
}

.risk-low    { color: #69f0ae; }
.risk-mod    { color: #ffb74d; }
.risk-high   { color: #ef5350; }

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.badge-low  { background:#1b5e20; color:#69f0ae; border:1px solid #69f0ae; }
.badge-mod  { background:#e65100; color:#ffb74d; border:1px solid #ffb74d; }
.badge-high { background:#b71c1c; color:#ef5350; border:1px solid #ef5350; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5c6bc0;
    border-left: 3px solid #5c6bc0;
    padding-left: 10px;
    margin: 24px 0 12px;
}

.info-box {
    background: #111827;
    border: 1px solid #1e2a4a;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.87rem;
    color: #90caf9;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── Helper: Load or train model ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Training Bayesian model…")
def load_or_train():
    model_path = DATA / "liver_model.joblib"

    # Generate ILPD.csv if missing
    ilpd_path = DATA / "ILPD.csv"
    if not ilpd_path.exists():
        exec(open(DATA / "generate_ilpd.py").read())

    if model_path.exists():
        obj = joblib.load(model_path)
        model      = obj["model"]
        boot_models = obj["boot_models"]
        feature_cols = obj["feature_cols"]
        combined_df = pd.read_csv(DATA / "Combined_dataset.csv") if (DATA / "Combined_dataset.csv").exists() else None
    else:
        from preprocess     import load_and_clean, save_cleaned
        from synthetic_data import generate_synthetic, combine_with_original
        from bmi            import add_bmi
        from model          import train

        df, scaler = load_and_clean(ilpd_path)
        save_cleaned(df, DATA / "Cleaned_data.csv")

        synth    = generate_synthetic(df, multiplier=6)
        synth.to_csv(DATA / "Synthetic_data.csv", index=False)

        combined = combine_with_original(df, synth)
        combined = add_bmi(combined)
        combined.to_csv(DATA / "Combined_dataset.csv", index=False)

        model, boot_models, feature_cols = train(combined, model_path=model_path)
        combined_df = combined

    return model, boot_models, feature_cols, combined_df


model, boot_models, feature_cols, combined_df = load_or_train()

# ══════════════════════════════════════════════════════════════════════════════
# ── Imports after sys.path set ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from bmi        import compute_bmi_single
from model      import predict_with_ci
from simulation import evolve_state, scenario_comparison, DEFAULT_SCENARIOS

# ══════════════════════════════════════════════════════════════════════════════
# ── HEADER ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center; padding: 32px 0 8px;'>
  <div style='font-family:"Space Mono",monospace; font-size:0.7rem; letter-spacing:0.25em;
              color:#5c6bc0; margin-bottom:10px;'>
    DIGITAL TWIN SIMULATOR
  </div>
  <h1 style='font-size:2.4rem; margin:0; background:linear-gradient(135deg,#90caf9,#ce93d8);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    Liver Disease Risk Engine
  </h1>
  <div style='color:#546e7a; font-size:0.9rem; margin-top:8px; font-family:"DM Sans",sans-serif;'>
    Obesity-induced NAFLD / Hepatitis risk · Bayesian CI · Time-evolution simulation
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR — Patient Inputs ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<div class='section-header'>Patient Profile</div>", unsafe_allow_html=True)

    age    = st.slider("Age (years)", 18, 85, 45, 1)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_code = 1 if gender == "Male" else 0

    st.markdown("<div class='section-header'>Anthropometrics</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", 35.0, 150.0, 80.0, 0.5)
    with col2:
        height = st.number_input("Height (m)", 1.40, 2.10, 1.68, 0.01)

    bmi_val, bmi_cat = compute_bmi_single(weight, height)
    bmi_color = "#69f0ae" if bmi_val < 25 else "#ffb74d" if bmi_val < 30 else "#ef5350"
    st.markdown(f"""
    <div class='metric-card' style='padding:12px;'>
      <div class='metric-value' style='font-size:1.8rem;color:{bmi_color};'>{bmi_val}</div>
      <div class='metric-label'>{bmi_cat}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Liver Panel</div>", unsafe_allow_html=True)
    total_bili  = st.slider("Total Bilirubin (mg/dL)", 0.1, 10.0, 1.2, 0.1)
    direct_bili = st.slider("Direct Bilirubin (mg/dL)", 0.1, 5.0, 0.4, 0.1)
    alkp        = st.slider("Alkaline Phosphatase (IU/L)", 44, 800, 200, 5)
    sgpt        = st.slider("SGPT / ALT (IU/L)", 7, 500, 45, 1)
    sgot        = st.slider("SGOT / AST (IU/L)", 10, 500, 40, 1)
    total_prot  = st.slider("Total Proteins (g/dL)", 2.7, 9.6, 6.8, 0.1)
    albumin     = st.slider("Albumin (g/dL)", 0.9, 5.5, 3.5, 0.1)
    ag_ratio    = st.slider("A/G Ratio", 0.3, 2.8, 1.1, 0.05)

    st.markdown("<div class='section-header'>Lifestyle</div>", unsafe_allow_html=True)
    exercise = st.slider("Exercise Level", 0.0, 1.0, 0.4, 0.05,
                         help="0 = sedentary · 1 = very active")
    alcohol  = st.slider("Alcohol Intake", 0.0, 1.0, 0.2, 0.05,
                         help="0 = none · 1 = heavy daily")
    diet     = st.slider("Diet Quality", 0.0, 1.0, 0.5, 0.05,
                         help="0 = poor · 1 = excellent")

    st.markdown("<div class='section-header'>Simulation</div>", unsafe_allow_html=True)
    sim_months = st.slider("Simulation horizon (months)", 6, 36, 18, 3)

# ══════════════════════════════════════════════════════════════════════════════
# ── Build patient dict ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

patient = {
    "age": age,
    "gender": gender_code,
    "bmi": bmi_val,
    "height_m": height,
    "weight_kg": weight,
    "total_bilirubin": total_bili,
    "direct_bilirubin": direct_bili,
    "alkaline_phosphotase": alkp,
    "sgpt": sgpt,
    "sgot": sgot,
    "total_proteins": total_prot,
    "albumin": albumin,
    "ag_ratio": ag_ratio,
    "exercise_level": exercise,
    "alcohol_intake": alcohol,
    "diet_quality": diet,
}

# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Dashboard",
    "⏱ Time Evolution",
    "🔬 Scenario Comparison",
    "🗃 Dataset Explorer",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Risk Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    result = predict_with_ci(patient, model, boot_models, feature_cols)
    prob   = result["mean_prob"]
    lo     = result["lower_ci"]
    hi     = result["upper_ci"]
    risk   = result["risk_level"]

    risk_class = {"Low":"low","Moderate":"mod","High":"high"}[risk]
    risk_color = {"Low":"#69f0ae","Moderate":"#ffb74d","High":"#ef5350"}[risk]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value' style='color:{risk_color};'>{prob:.1%}</div>
          <div class='metric-label'>Disease Probability</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value' style='color:#90caf9; font-size:1.6rem;'>{lo:.1%} – {hi:.1%}</div>
          <div class='metric-label'>90% Confidence Interval</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value' style='color:{bmi_color};'>{bmi_val}</div>
          <div class='metric-label'>BMI · {bmi_cat}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
          <span class='badge badge-{risk_class}'>{risk.upper()} RISK</span>
          <div class='metric-label' style='margin-top:10px;'>Overall Assessment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_gauge, col_radar = st.columns([1, 1])

    # ── Gauge ───────────────────────────────────────────────────────────────
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={"reference": 50, "valueformat": ".1f", "suffix": "%"},
            title={"text": "Liver Disease Risk", "font": {"color": "#e8eaf6", "family": "Space Mono"}},
            number={"suffix": "%", "font": {"color": risk_color, "size": 48, "family": "Space Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#546e7a",
                         "tickfont": {"color": "#546e7a"}},
                "bar": {"color": risk_color, "thickness": 0.25},
                "bgcolor": "#1a1f35",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35],  "color": "#1b3a1e"},
                    {"range": [35, 65], "color": "#3e2723"},
                    {"range": [65, 100],"color": "#4a1a1a"},
                ],
                "threshold": {
                    "line": {"color": "#ef5350", "width": 2},
                    "thickness": 0.75,
                    "value": hi * 100
                },
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0a0e1a", font_color="#e8eaf6",
            height=320, margin=dict(t=60, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # CI band annotation
        st.markdown(f"""
        <div class='info-box'>
          <strong>Prediction CI</strong><br/>
          90% of bootstrap models predict between
          <strong style='color:#90caf9;'>{lo:.1%}</strong> and
          <strong style='color:#90caf9;'>{hi:.1%}</strong> — width = {(hi-lo):.1%}
        </div>""", unsafe_allow_html=True)

    # ── Radar / feature contribution ────────────────────────────────────────
    with col_radar:
        labels = ["BMI Risk", "Enzyme Stress", "Bilirubin", "Albumin (inv)", "Lifestyle Risk"]

        # Normalise to 0–1 risk contribution
        bmi_risk     = np.clip((bmi_val - 18.5) / 21.5, 0, 1)
        enzyme_stress= np.clip((np.log1p(sgpt) + np.log1p(sgot)) / 14, 0, 1)
        bili_risk    = np.clip(total_bili / 10, 0, 1)
        alb_inv      = 1 - np.clip((albumin - 0.9) / 4.6, 0, 1)
        life_risk    = np.clip(alcohol * 0.5 - diet * 0.25 - exercise * 0.25 + 0.3, 0, 1)

        values = [bmi_risk, enzyme_stress, bili_risk, alb_inv, life_risk]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor=f"rgba(239,83,80,0.15)",
            line=dict(color=risk_color, width=2),
            name="Risk Profile"
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#111827",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#546e7a"),
                                gridcolor="#1e2a4a"),
                angularaxis=dict(tickfont=dict(color="#90caf9", family="Space Mono", size=11),
                                 gridcolor="#1e2a4a"),
            ),
            paper_bgcolor="#0a0e1a", font_color="#e8eaf6",
            showlegend=False, height=320,
            margin=dict(t=40, b=20, l=60, r=60),
            title=dict(text="Risk Factor Breakdown", font=dict(family="Space Mono", color="#e8eaf6"))
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Clinical markers table ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Clinical Markers Summary</div>", unsafe_allow_html=True)
    markers = pd.DataFrame({
        "Marker": ["Total Bilirubin","Direct Bilirubin","Alkaline Phosphatase","SGPT","SGOT",
                   "Total Proteins","Albumin","A/G Ratio","BMI"],
        "Value":  [total_bili, direct_bili, alkp, sgpt, sgot, total_prot, albumin, ag_ratio, bmi_val],
        "Unit":   ["mg/dL","mg/dL","IU/L","IU/L","IU/L","g/dL","g/dL","ratio","kg/m²"],
        "Normal Range": ["0.2–1.2","0.1–0.4","44–147","7–56","10–40","6.3–8.2","3.5–5.0","1.0–2.5","18.5–24.9"],
    })
    st.dataframe(markers, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Time Evolution (Current Lifestyle)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Digital Twin — Risk Evolution Over Time")
    st.markdown(f"""
    <div class='info-box'>
      Simulating <strong>{sim_months} months</strong> with your current lifestyle inputs.
      Each month updates BMI, liver enzymes, and bilirubin using physiological rules.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running simulation…"):
        evo_df = evolve_state(
            patient, exercise=exercise, alcohol=alcohol, diet=diet,
            months=sim_months, model=model, boot_models=boot_models, feature_cols=feature_cols
        )

    if "mean_prob" not in evo_df.columns:
        st.error("Prediction unavailable — please check model load.")
    else:
        fig_evo = go.Figure()

        # CI band
        fig_evo.add_trace(go.Scatter(
            x=list(evo_df["month"]) + list(evo_df["month"])[::-1],
            y=list(evo_df["upper_ci"]) + list(evo_df["lower_ci"])[::-1],
            fill="toself",
            fillcolor="rgba(92,107,192,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI",
        ))
        # Mean probability
        fig_evo.add_trace(go.Scatter(
            x=evo_df["month"], y=evo_df["mean_prob"],
            mode="lines+markers",
            line=dict(color="#90caf9", width=2.5),
            marker=dict(size=5),
            name="Disease Probability",
        ))
        # 35% risk threshold
        fig_evo.add_hline(y=0.35, line_dash="dot", line_color="#69f0ae", opacity=0.5,
                          annotation_text="Low/Moderate threshold", annotation_position="right")
        fig_evo.add_hline(y=0.65, line_dash="dot", line_color="#ef5350", opacity=0.5,
                          annotation_text="High risk threshold", annotation_position="right")

        fig_evo.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
            font_color="#e8eaf6",
            xaxis=dict(title="Month", gridcolor="#1e2a4a", title_font_color="#546e7a"),
            yaxis=dict(title="Liver Disease Probability", tickformat=".0%",
                       range=[0, 1], gridcolor="#1e2a4a", title_font_color="#546e7a"),
            legend=dict(bgcolor="#111827", bordercolor="#1e2a4a"),
            title=dict(text="Risk Probability Over Time", font=dict(family="Space Mono")),
            height=420, margin=dict(t=60, b=60, l=60, r=40),
        )
        st.plotly_chart(fig_evo, use_container_width=True)

        # Secondary biomarker chart
        fig_bio = make_subplots(rows=2, cols=2,
                                subplot_titles=["BMI", "SGPT (ALT)", "SGOT (AST)", "Albumin"])
        for trace, row, col in [
            (go.Scatter(x=evo_df["month"], y=evo_df["bmi"],     mode="lines",
                        line=dict(color="#ce93d8"), name="BMI"),   1, 1),
            (go.Scatter(x=evo_df["month"], y=evo_df["sgpt"],    mode="lines",
                        line=dict(color="#ffb74d"), name="SGPT"),  1, 2),
            (go.Scatter(x=evo_df["month"], y=evo_df["sgot"],    mode="lines",
                        line=dict(color="#ef9a9a"), name="SGOT"),  2, 1),
            (go.Scatter(x=evo_df["month"], y=evo_df["albumin"], mode="lines",
                        line=dict(color="#80cbc4"), name="Albumin"), 2, 2),
        ]:
            fig_bio.add_trace(trace, row=row, col=col)

        fig_bio.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
            font_color="#e8eaf6", showlegend=False, height=380,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        for ax in fig_bio.layout:
            if "xaxis" in ax or "yaxis" in ax:
                fig_bio.layout[ax].update(gridcolor="#1e2a4a")

        st.plotly_chart(fig_bio, use_container_width=True)

        # Summary table
        st.markdown("<div class='section-header'>Monthly Snapshot</div>", unsafe_allow_html=True)
        display_df = evo_df[["month","bmi","sgpt","sgot","albumin",
                              "total_bilirubin","mean_prob","lower_ci","upper_ci"]].copy()
        display_df.columns = ["Month","BMI","SGPT","SGOT","Albumin","T.Bili","P(Disease)","CI Low","CI High"]
        display_df = display_df[::3]   # show every 3 months
        st.dataframe(display_df.style.format({
            "BMI":"{:.1f}", "SGPT":"{:.0f}", "SGOT":"{:.0f}",
            "Albumin":"{:.2f}", "T.Bili":"{:.2f}",
            "P(Disease)":"{:.1%}", "CI Low":"{:.1%}", "CI High":"{:.1%}"
        }), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Scenario Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Lifestyle Scenario Comparison")

    # Custom scenarios from current sidebar values
    scenarios = {
        "Current Lifestyle":   {"exercise": exercise, "alcohol": alcohol, "diet": diet},
        "Moderate Improvement":{"exercise": min(exercise + 0.3, 1.0),
                                "alcohol":  max(alcohol - 0.3, 0.0),
                                "diet":     min(diet + 0.3, 1.0)},
        "Optimal Lifestyle":   {"exercise": 0.9, "alcohol": 0.05, "diet": 0.9},
        "Worst Case":          {"exercise": 0.0, "alcohol": 1.0,  "diet": 0.0},
    }

    with st.spinner("Comparing scenarios…"):
        scenario_results = scenario_comparison(
            patient, scenarios, months=sim_months,
            model=model, boot_models=boot_models, feature_cols=feature_cols
        )

    COLORS = {"Current Lifestyle":"#90caf9","Moderate Improvement":"#ffb74d",
              "Optimal Lifestyle":"#69f0ae","Worst Case":"#ef5350"}

    fig_comp = go.Figure()
    for name, df_s in scenario_results.items():
        if "mean_prob" not in df_s.columns:
            continue
        color = COLORS.get(name, "#ffffff")
        fig_comp.add_trace(go.Scatter(
            x=df_s["month"], y=df_s["mean_prob"],
            mode="lines", name=name,
            line=dict(color=color, width=2.5 if name == "Current Lifestyle" else 1.8,
                      dash="solid" if name != "Worst Case" else "dash"),
        ))

    fig_comp.add_hline(y=0.65, line_dash="dot", line_color="#ef5350", opacity=0.4)
    fig_comp.add_hline(y=0.35, line_dash="dot", line_color="#69f0ae", opacity=0.4)
    fig_comp.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font_color="#e8eaf6",
        xaxis=dict(title="Month", gridcolor="#1e2a4a"),
        yaxis=dict(title="Disease Probability", tickformat=".0%", range=[0, 1], gridcolor="#1e2a4a"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2a4a", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Risk Trajectory by Lifestyle Scenario", font=dict(family="Space Mono")),
        height=430, margin=dict(t=80, b=60, l=60, r=40),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # End-state summary
    st.markdown("<div class='section-header'>End-State Comparison (Month {})".format(sim_months) +
                "</div>", unsafe_allow_html=True)
    rows = []
    for name, df_s in scenario_results.items():
        if "mean_prob" not in df_s.columns:
            continue
        last = df_s.iloc[-1]
        rows.append({
            "Scenario":     name,
            "Final Risk":   f"{last['mean_prob']:.1%}",
            "Risk Level":   last.get("risk_level", "N/A"),
            "Final BMI":    f"{last['bmi']:.1f}",
            "SGPT":         f"{last['sgpt']:.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Lifestyle parameter table
    st.markdown("<div class='section-header'>Scenario Parameters</div>", unsafe_allow_html=True)
    param_rows = []
    for name, params in scenarios.items():
        param_rows.append({
            "Scenario":       name,
            "Exercise Level": f"{params['exercise']:.2f}",
            "Alcohol Intake": f"{params['alcohol']:.2f}",
            "Diet Quality":   f"{params['diet']:.2f}",
        })
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Dataset Explorer")

    if combined_df is not None:
        df_show = combined_df.copy()

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Patients", len(df_show))
        with c2: st.metric("Disease Cases", int(df_show["liver_disease"].sum()))
        with c3: st.metric("Synthetic Records",
                           int(df_show["synthetic"].sum()) if "synthetic" in df_show.columns else "N/A")

        # Distribution plots
        fig_dist = make_subplots(rows=1, cols=3,
                                 subplot_titles=["Age Distribution","BMI Distribution","SGPT Distribution"])
        for col, row_i, col_i, color in [
            ("age",  1, 1, "#90caf9"),
            ("bmi",  1, 2, "#ce93d8"),
            ("sgpt", 1, 3, "#ffb74d"),
        ]:
            if col not in df_show.columns:
                continue
            for ld, name, opacity in [(1, "Disease", 0.7), (0, "Healthy", 0.5)]:
                vals = df_show.loc[df_show["liver_disease"] == ld, col].dropna()
                fig_dist.add_trace(
                    go.Histogram(x=vals, name=name, opacity=opacity,
                                 marker_color="#ef5350" if ld==1 else "#69f0ae",
                                 showlegend=(col == "age")),
                    row=1, col=col_i
                )

        fig_dist.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
            font_color="#e8eaf6", barmode="overlay", height=320,
            margin=dict(t=50, b=40, l=40, r=20),
            legend=dict(bgcolor="#111827"),
        )
        for ax in fig_dist.layout:
            if "xaxis" in ax or "yaxis" in ax:
                fig_dist.layout[ax].update(gridcolor="#1e2a4a")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Correlation heat
        num_cols = ["age","bmi","sgpt","sgot","total_bilirubin","albumin",
                    "liver_disease","exercise_level","alcohol_intake","diet_quality"]
        num_cols = [c for c in num_cols if c in df_show.columns]
        corr = df_show[num_cols].corr()

        fig_heat = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Feature Correlation Matrix",
        )
        fig_heat.update_layout(
            paper_bgcolor="#0a0e1a", font_color="#e8eaf6",
            title_font_family="Space Mono", height=500,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Raw table sample
        st.markdown("<div class='section-header'>Sample Records (50 rows)</div>", unsafe_allow_html=True)
        st.dataframe(df_show.head(50), use_container_width=True)

        # Download
        csv_bytes = df_show.to_csv(index=False).encode()
        st.download_button("⬇ Download Combined Dataset", csv_bytes,
                           "Combined_dataset.csv", "text/csv")
    else:
        st.warning("Combined dataset not found. Run the pipeline to generate it.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:40px 0 16px; color:#37474f; font-size:0.78rem;
            font-family:"Space Mono",monospace;'>
  Digital Twin Liver Risk Engine · IIT BHU Hackathon 2025 · Powered by Bayesian ML
</div>
""", unsafe_allow_html=True)
