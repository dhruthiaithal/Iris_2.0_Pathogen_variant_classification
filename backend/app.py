import os
import subprocess
import json
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

# Serve uploaded CSV files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# -------------------------
# Load ML Model
# -------------------------

xgb_model = XGBClassifier()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

xgb_model.load_model(
    os.path.join(MODEL_DIR, "xgb_pathogenicity_model1.json")
)

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
# Breast Cancer Gene Definitions
# -------------------------

CORE_GENES = ["BRCA1", "BRCA2"]

HR_PATHWAY_GENES = [
    "RAD51", "PALB2", "ATM", "CHEK2",
    "BARD1", "BRIP1", "NBN", "RAD50", "MRE11",
    "CHEK1", "TP53", "PTEN", "STK11",
    "CDK12", "FANCD2", "FANCA", "FANCL",
]

ALL_BREAST_CANCER_GENES = set(CORE_GENES + HR_PATHWAY_GENES)

# -------------------------
# Variant consequence severity
# -------------------------

CONSEQUENCE_SEVERITY = {
    "frameshift deletion": 1.0,
    "frameshift insertion": 1.0,
    "stopgain": 0.95,
    "stoploss": 0.90,
    "splicing": 0.90,
    "nonsynonymous SNV": 0.50,
    "nonframeshift deletion": 0.40,
    "nonframeshift insertion": 0.40,
    "synonymous SNV": 0.05,
    "unknown": 0.10,
}

# -------------------------
# Gene risk weights
# -------------------------

GENE_RISK_WEIGHTS = {
    "BRCA1": 1.00,
    "BRCA2": 0.95,
    "PALB2": 0.75,
    "TP53": 0.80,
    "PTEN": 0.75,
    "STK11": 0.70,
    "ATM": 0.55,
    "CHEK2": 0.50,
    "BARD1": 0.45,
    "BRIP1": 0.40,
    "NBN": 0.40,
    "RAD50": 0.40,
    "MRE11": 0.35,
    "CHEK1": 0.35,
    "RAD51": 0.35,
    "CDK12": 0.25,
    "FANCD2": 0.25,
    "FANCA": 0.20,
    "FANCL": 0.20,
}

POPULATION_AF_THRESHOLD = 0.01

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
        "-protocol",
        "refGene,gnomad211_exome,dbnsfp30a,clinvar_20250721",
        "-operation", "g,f,f,f",
        "-nastring", ".",
        "-vcfinput"
    ]

    subprocess.run(cmd, check=True)

# -------------------------
# Genetic Risk Score
# -------------------------

def calculate_genetic_risk_score(
    df,
    probs,
    gene_col="Gene.refGene",
    consequence_col="ExonicFunc.refGene",
    af_col="AF"
):

    contributing_variants = []
    complement_product = 1.0

    for i in range(len(df)):

        row = df.iloc[i]

        gene_raw = str(row.get(gene_col, ".")).strip()

        genes_in_row = [
            g.strip()
            for g in gene_raw.replace(",", ";").split(";")
        ]

        matched_genes = [
            g for g in genes_in_row
            if g in ALL_BREAST_CANCER_GENES
        ]

        if not matched_genes:
            continue

        af_value = row.get(af_col, ".")

        try:
            af_float = float(af_value)

            if af_float > POPULATION_AF_THRESHOLD:
                continue

        except (ValueError, TypeError):
            pass

        consequence_raw = str(
            row.get(consequence_col, "unknown")
        ).strip().lower()

        consequence_weight = next(
            (
                v for k, v in CONSEQUENCE_SEVERITY.items()
                if k in consequence_raw
            ),
            CONSEQUENCE_SEVERITY["unknown"]
        )

        if consequence_weight < 0.10:
            continue

        path_prob = float(probs[i])

        if path_prob < 0.10:
            continue

        best_gene = max(
            matched_genes,
            key=lambda g: GENE_RISK_WEIGHTS.get(g, 0)
        )

        gene_weight = GENE_RISK_WEIGHTS.get(best_gene, 0.10)

        variant_score = (
            gene_weight *
            consequence_weight *
            path_prob
        )

        variant_score = min(variant_score, 1.0)

        complement_product *= (1.0 - variant_score)

        contributing_variants.append({
            "variant_index": int(i),
            "gene": best_gene,
            "consequence": str(
                row.get(consequence_col, ".")
            ),
            "pathogenicity_prob": round(path_prob, 4),
            "gene_weight": gene_weight,
            "consequence_weight": consequence_weight,
            "variant_score": round(variant_score, 4),
        })

    grs = round(1.0 - complement_product, 4)

    if grs < 0.20:
        risk_tier = "Low Risk"

        interpretation = (
            "No or minimal pathogenic variants detected."
        )

    elif grs < 0.50:
        risk_tier = "Moderate Risk"

        interpretation = (
            "Variants detected in moderate-risk genes."
        )

    elif grs < 0.75:
        risk_tier = "High Risk"

        interpretation = (
            "Pathogenic variants detected in high-risk genes."
        )

    else:
        risk_tier = "Very High Risk"

        interpretation = (
            "Strong hereditary breast cancer signal detected."
        )

    return {
        "grs": grs,
        "risk_tier": risk_tier,
        "contributing_variants": contributing_variants,
        "interpretation": interpretation,
    }

# -------------------------
# API Health Check
# -------------------------

@app.get("/")
def home():
    return {
        "message": "Variant Pathogenicity API is running"
    }

# -------------------------
# Prediction Endpoint
# -------------------------

@app.post("/predict")
async def predict_variant(file: UploadFile = File(...)):

    # Save uploaded file
    vcf_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(vcf_path, "wb") as f:
        f.write(await file.read())

    # Run ANNOVAR
    out_prefix = f"{UPLOAD_DIR}/annotated"

    run_annovar(vcf_path, out_prefix)

    annotated_file = f"{out_prefix}.hg38_multianno.txt"

    # Load ANNOVAR output
    df = pd.read_csv(annotated_file, sep="\t")

    # Feature mapping
    feature_map = {
        "CADD_score": "CADD_phred",
        "SIFT_score": "SIFT_score",
        "PolyPhen_score": "Polyphen2_HDIV_score",
        "gnomAD_AF": "AF",
        "Conservation_score": "GERP++_RS",
        "Consequence_encoded": "ExonicFunc.refGene"
    }

    for model_feature, annovar_feature in feature_map.items():

        df[model_feature] = (
            df[annovar_feature]
            if annovar_feature in df.columns
            else np.nan
        )

    # Derived features
    df["SIFT_missing"] = (
        df["SIFT_score"].isna().astype(int)
    )

    df["PolyPhen_missing"] = (
        df["PolyPhen_score"].isna().astype(int)
    )

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

    df["Consequence_encoded"] = (
        df["Consequence_encoded"]
        .map(consequence_map)
        .fillna(0)
    )

    # Ensure features exist
    missing_features = []

    for feature in FEATURES:

        if feature not in df.columns:
            df[feature] = np.nan
            missing_features.append(feature)

    # Prepare input
    X = (
        df[FEATURES]
        .replace(".", np.nan)
        .apply(pd.to_numeric, errors="coerce")
    )

    predictor_columns = [
        "CADD_score",
        "SIFT_score",
        "PolyPhen_score",
        "gnomAD_AF",
        "Conservation_score"
    ]

    missing_predictors_mask = (
        X[predictor_columns]
        .isna()
        .all(axis=1)
    )

    X = X.fillna(X.median()).fillna(0)

    # Predictions
    probs = xgb_model.predict_proba(X)[:, 1]

    labels = [
        "Pathogenic" if p >= 0.5 else "Benign"
        for p in probs
    ]

    # SHAP
    shap_values = explainer.shap_values(X)

    # GRS
    grs_result = calculate_genetic_risk_score(
        df=df,
        probs=probs,
        gene_col="Gene.refGene",
        consequence_col="ExonicFunc.refGene",
        af_col="AF",
    )

    # Results
    results = []

    for i in range(len(X)):

        if missing_predictors_mask.iloc[i]:

            prediction = "Uncertain Significance"
            confidence = 0
            explanation = {}

        else:

            prediction = labels[i]

            prob = float(probs[i])

            confidence = (
                round(prob * 100, 2)
                if prediction == "Pathogenic"
                else round((1 - prob) * 100, 2)
            )

            explanation = dict(
                zip(FEATURES, shap_values[i].tolist())
            )

            explanation = dict(
                sorted(
                    explanation.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
            )

        clinvar_disease = ""

        if prediction == "Pathogenic" and "CLNDN" in df.columns:

            disease = df.iloc[i]["CLNDN"]

            if pd.notna(disease) and disease != ".":

                clinvar_disease = (
                    disease.split("|")[0]
                    .replace("_", " ")
                )

        variant_result = {
            "variant_index": int(i),

            "gene": str(
                df.iloc[i].get("Gene.refGene", ".")
            ),

            "consequence": str(
                df.iloc[i].get(
                    "ExonicFunc.refGene",
                    "."
                )
            ),

            "prediction": prediction,

            "confidence": confidence,

            "pathogenicity_probability": round(
                float(probs[i]), 4
            ),

            "explanation": explanation,

            "clinvar_disease": clinvar_disease
        }

        for col in ["Chr", "Start", "End", "Ref", "Alt"]:

            if col in df.columns:

                value = df.iloc[i][col]

                if isinstance(value, (np.integer, np.int64)):
                    value = int(value)

                elif isinstance(
                    value,
                    (np.floating, np.float64)
                ):
                    value = float(value)

                variant_result[col] = value

        # GRS contribution
        variant_result["grs_contribution"] = 0

        for v in grs_result["contributing_variants"]:

            if v["variant_index"] == i:

                variant_result["grs_contribution"] = (
                    v["variant_score"]
                )

                break

        results.append(variant_result)

    # Important variants CSV
    important_variants = []

    for r in results:

        if (
            r["prediction"] == "Pathogenic"
            or r["grs_contribution"] >= 0.20
            or r["clinvar_disease"]
        ):

            important_variants.append({
                "Chr": r.get("Chr", ""),
                "Position": r.get("Start", ""),
                "Ref": r.get("Ref", ""),
                "Alt": r.get("Alt", ""),
                "Gene": r.get("gene", ""),
                "Consequence": r.get("consequence", ""),
                "Prediction": r.get("prediction", ""),
                "Confidence": r.get("confidence", ""),
                "Pathogenicity_Probability":
                    r.get("pathogenicity_probability", ""),
                "ClinVar_Disease":
                    r.get("clinvar_disease", ""),
                "GRS_Contribution":
                    r.get("grs_contribution", ""),
                "Overall_GRS":
                    grs_result["grs"],
                "Risk_Tier":
                    grs_result["risk_tier"],
            })

    important_df = pd.DataFrame(important_variants)

    csv_output_path = (
        f"{UPLOAD_DIR}/important_variants.csv"
    )

    important_df.to_csv(csv_output_path, index=False)

    return {
        "total_variants": len(results),

        "results": results,

        "missing_features": missing_features,

        "breast_cancer_genetic_risk": grs_result,

        "important_variants_csv":
            "http://127.0.0.1:8000/uploads/important_variants.csv"
    }