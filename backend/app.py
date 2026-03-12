import os
import subprocess
import json
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, UploadFile, File
from xgboost import XGBClassifier
# import tensorflow as tf   # Neural network disabled for now

UPLOAD_DIR = "uploads"
ANNOVAR_DIR = "/root/annovar"
HUMANDB = f"{ANNOVAR_DIR}/humandb"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# -------------------------
# Load XGBoost model
# -------------------------
xgb_model = XGBClassifier()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_pathogenicity_model.json"))

# Neural network temporarily disabled
# nn_model = tf.keras.models.load_model(
#     os.path.join(MODEL_DIR, "nn_pathogenicity_model.keras")
# )

# Load feature list
with open(os.path.join(MODEL_DIR, "features.json")) as f:
    FEATURES = json.load(f)

# SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# -------------------------
# Utility: Run ANNOVAR
# -------------------------
def run_annovar(vcf_path, output_prefix):
    cmd = [
        "perl", f"{ANNOVAR_DIR}/table_annovar.pl",
        vcf_path,
        HUMANDB,
        "-buildver", "hg19",
        "-out", output_prefix,
        "-remove",
        "-protocol", "refGene",
        "-operation", "g",
        "-nastring", ".",
        "-vcfinput"
    ]
    subprocess.run(cmd, check=True)

@app.get("/")
def home():
    return {"message": "Variant Pathogenicity API is running"}
# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
async def predict_variant(file: UploadFile = File(...)):

    # Save uploaded VCF
    vcf_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(vcf_path, "wb") as f:
        f.write(await file.read())

    # Run ANNOVAR annotation
    out_prefix = f"{UPLOAD_DIR}/annotated"
    run_annovar(vcf_path, out_prefix)

    # Load annotated variant file
    df = pd.read_csv(f"{out_prefix}.hg19_multianno.txt", sep="\t")

    # Extract ML features
    X = df[FEATURES].replace(".", np.nan).astype(float)
    X = X.fillna(X.median())

    # -------------------------
    # Prediction (XGBoost only)
    # -------------------------
    xgb_prob = float(xgb_model.predict_proba(X)[0, 1])

    # Neural network removed
    # nn_prob = float(nn_model.predict(X)[0][0])
    # final_prob = (xgb_prob + nn_prob) / 2

    final_prob = xgb_prob

    label = "Pathogenic" if final_prob >= 0.5 else "Benign"

    # -------------------------
    # SHAP Explanation
    # -------------------------
    shap_values = explainer.shap_values(X)

    explanation = dict(
        zip(FEATURES, shap_values[0].tolist())
    )

    return {
        "prediction": label,
        "confidence": round(final_prob, 3),
        "model_breakdown": {
            "xgboost": round(xgb_prob, 3)
        },
        "explanation": explanation
    }