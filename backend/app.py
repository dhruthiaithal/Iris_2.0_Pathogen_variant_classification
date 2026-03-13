import os
import subprocess
import json
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier

# -------------------------
# Initialize API
# -------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Paths
# -------------------------

UPLOAD_DIR = "uploads"
ANNOVAR_DIR = "/root/annovar"
HUMANDB = f"{ANNOVAR_DIR}/humandb"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Load ML Model
# -------------------------

xgb_model = XGBClassifier()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_pathogenicity_model1.json"))

# -------------------------
# Load feature list
# -------------------------

with open(os.path.join(MODEL_DIR, "features.json")) as f:
    FEATURES = json.load(f)

# -------------------------
# SHAP Explainer
# -------------------------

explainer = shap.TreeExplainer(xgb_model)

# -------------------------
# Run ANNOVAR
# -------------------------

def run_annovar(vcf_path, output_prefix):
    cmd = [
        "perl",
        f"{ANNOVAR_DIR}/table_annovar.pl",
        vcf_path,
        HUMANDB,
        "-buildver", "hg38",
        "-out", output_prefix,
        "-remove",
        "-protocol", "refGene,gnomad211_exome,dbnsfp30a,clinvar_20250721",
        "-operation", "g,f,f,f",
        "-nastring", ".",
        "-vcfinput"
    ]
    subprocess.run(cmd, check=True)

# -------------------------
# API Health Check
# -------------------------

@app.get("/")
def home():
    return {"message": "Variant Pathogenicity API is running"}

# -------------------------
# Prediction Endpoint
# -------------------------

@app.post("/predict")
async def predict_variant(file: UploadFile = File(...)):

    # -------------------------
    # Save uploaded VCF
    # -------------------------

    vcf_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(vcf_path, "wb") as f:
        f.write(await file.read())

    # -------------------------
    # Run ANNOVAR
    # -------------------------

    out_prefix = f"{UPLOAD_DIR}/annotated"
    run_annovar(vcf_path, out_prefix)
    annotated_file = f"{out_prefix}.hg38_multianno.txt"

    # -------------------------
    # Load ANNOVAR output
    # -------------------------

    df = pd.read_csv(annotated_file, sep="\t")

    # -------------------------
    # Map ANNOVAR columns to ML features
    # -------------------------

    feature_map = {
        "CADD_score": "CADD_phred",
        "SIFT_score": "SIFT_score",
        "PolyPhen_score": "Polyphen2_HDIV_score",
        "gnomAD_AF": "AF",
        "Conservation_score": "GERP++_RS",
        "Consequence_encoded": "ExonicFunc.refGene"
    }

    for model_feature, annovar_feature in feature_map.items():
        df[model_feature] = df[annovar_feature] if annovar_feature in df.columns else np.nan

    # -------------------------
    # Derived features
    # -------------------------

    df["SIFT_missing"] = df["SIFT_score"].isna().astype(int)
    df["PolyPhen_missing"] = df["PolyPhen_score"].isna().astype(int)

    consequence_map = {
        "synonymous SNV": 0,
        "nonsynonymous SNV": 1,
        "stopgain": 2,
        "stoploss": 3,
        "frameshift deletion": 4,
        "frameshift insertion": 4,
        "nonframeshift deletion": 5,
        "nonframeshift insertion": 5
    }
    df["Consequence_encoded"] = df["Consequence_encoded"].map(consequence_map).fillna(0)

    # -------------------------
    # Ensure all model features exist
    # -------------------------

    missing_features = []
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan
            missing_features.append(feature)

    # -------------------------
    # Prepare ML input
    # -------------------------

    X = df[FEATURES].replace(".", np.nan).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)

    # -------------------------
    # Model Predictions
    # -------------------------

    probs = xgb_model.predict_proba(X)[:, 1]
    labels = ["Pathogenic" if p >= 0.5 else "Benign" for p in probs]

    # -------------------------
    # SHAP Explanation
    # -------------------------

    shap_values = explainer.shap_values(X)
    results = []

    for i in range(len(X)):

        # Top 10 SHAP features
        explanation = dict(zip(FEATURES, shap_values[i].tolist()))
        explanation = dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

        # -------------------------
        # ClinVar disease extraction (FIXED)
        # -------------------------

        clinvar_disease = ""

        if labels[i] == "Pathogenic" and "CLNDN" in df.columns:
            disease = df.iloc[i]["CLNDN"]
            if pd.notna(disease) and disease != ".":
                clinvar_disease = disease.split("|")[0].replace("_", " ")

        prob = float(probs[i])

        if labels[i] == "Pathogenic":
            confidence = round(prob * 100, 2)
        else:
            confidence = round((1 - prob) * 100, 2)

        variant_result = {
            "variant_index": int(i),
            "prediction": labels[i],
            "confidence": confidence,
            "explanation": explanation,
            "clinvar_disease": clinvar_disease
        }

        # Genomic coordinates
        for col in ["Chr", "Start", "End", "Ref", "Alt"]:
            if col in df.columns:
                value = df.iloc[i][col]
                if isinstance(value, (np.integer, np.int64)):
                    value = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    value = float(value)
                variant_result[col] = value

        results.append(variant_result)

    # -------------------------
    # Return API response
    # -------------------------

    return {
        "total_variants": len(results),
        "results": results,
        "missing_features": missing_features
    }