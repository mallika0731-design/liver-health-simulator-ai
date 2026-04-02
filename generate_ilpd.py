"""
Generate a realistic ILPD (Indian Liver Patient Dataset) CSV
with 416 rows matching the original dataset structure.
Used when the actual ILPD.csv is not present.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

n = 416
n_disease = 295   # ~71% have liver disease (original ratio)
n_healthy = 121

def gen_group(n, disease=True):
    rows = []
    for _ in range(n):
        age = int(np.clip(np.random.normal(45 if disease else 38, 13), 4, 90))
        gender = np.random.choice(["Male", "Female"], p=[0.75, 0.25])
        if disease:
            tb  = np.clip(np.random.lognormal(1.1, 0.8), 0.4, 75)
            db  = np.clip(tb * np.random.uniform(0.3, 0.7), 0.1, 40)
            alkp = np.clip(np.random.lognormal(5.4, 0.6), 60, 2110)
            sgpt = np.clip(np.random.lognormal(4.2, 1.0), 7, 2000)
            sgot = np.clip(np.random.lognormal(4.0, 1.0), 10, 4929)
            tp   = np.clip(np.random.normal(6.2, 1.1), 2.7, 9.6)
            alb  = np.clip(np.random.normal(2.8, 0.7), 0.9, 5.5)
        else:
            tb  = np.clip(np.random.lognormal(0.2, 0.3), 0.4, 3)
            db  = np.clip(tb * np.random.uniform(0.15, 0.35), 0.1, 1)
            alkp = np.clip(np.random.lognormal(4.8, 0.3), 63, 300)
            sgpt = np.clip(np.random.lognormal(3.2, 0.5), 7, 100)
            sgot = np.clip(np.random.lognormal(3.1, 0.4), 10, 130)
            tp   = np.clip(np.random.normal(7.0, 0.8), 4.0, 9.6)
            alb  = np.clip(np.random.normal(3.6, 0.5), 2.0, 5.5)
        agr = np.clip(alb / max(tp - alb, 0.1), 0.3, 2.8)
        rows.append([age, gender, round(tb,1), round(db,1), int(alkp),
                     int(sgpt), int(sgot), round(tp,1), round(alb,1),
                     round(agr,2), 1 if disease else 2])
    return rows

data = gen_group(n_disease, True) + gen_group(n_healthy, False)
np.random.shuffle(data)

cols = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin",
        "Alkaline_Phosphotase","Alamine_Aminotransferase",
        "Aspartate_Aminotransferase","Total_Protiens",
        "Albumin","Albumin_and_Globulin_Ratio","Dataset"]

df = pd.DataFrame(data, columns=cols)
df.to_csv("/home/claude/digital_twin_liver/data/ILPD.csv", index=False)
print(f"Generated ILPD.csv with {len(df)} rows")
print(df["Dataset"].value_counts())
