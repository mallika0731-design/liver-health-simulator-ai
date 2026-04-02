# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Digital Twin Liver Disease Risk Simulator
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Digital Twin Team"
LABEL description="Obesity-Induced Liver Disease Risk · Digital Twin Simulator"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Pre-generate synthetic data and train model on build
# (speeds up first Streamlit load significantly)
RUN python -c "
import sys, os
sys.path.insert(0, 'src')
os.makedirs('data', exist_ok=True)

# Generate ILPD
exec(open('data/generate_ilpd.py').read())

from preprocess     import load_and_clean, save_cleaned
from synthetic_data import generate_synthetic, combine_with_original
from bmi            import add_bmi
from model          import train

df, _ = load_and_clean('data/ILPD.csv')
save_cleaned(df, 'data/Cleaned_data.csv')
synth = generate_synthetic(df, multiplier=6)
synth.to_csv('data/Synthetic_data.csv', index=False)
comb = combine_with_original(df, synth)
comb = add_bmi(comb)
comb.to_csv('data/Combined_dataset.csv', index=False)
train(comb, model_path='data/liver_model.joblib')
print('Pre-training complete')
"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"]
