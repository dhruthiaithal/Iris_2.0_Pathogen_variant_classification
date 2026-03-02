from google.colab import files
uploaded = files.upload() #upload csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("gnomAD_1.csv")

# Basic inspection
print(df.shape)
print(df.head())
print(df.isna().sum())
df = df.dropna(how="all")
df = df.reset_index(drop=True)
clinvar_map = {
    "Benign": 0,
    "Likely benign": 0,
    "Benign/Likely benign": 0,
    "Pathogenic": 1,
    "Likely pathogenic": 1,
    "Pathogenic/Likely pathogenic": 1
    
}

df["Label"] = df["ClinVar Germline Classification"].map(clinvar_map)

# Drop unlabelled / ambiguous variants
df = df.dropna(subset=["Label"])
df["Label"] = df["Label"].astype(int)

print(df["Label"].value_counts())
FEATURES = [
    "cadd",
    "Allele Frequency",      # gnomAD AF
    "sift_max",
    "polyphen_max",
    "phylop",
    "Transcript Consequence"
]
df = df.rename(columns={
    "cadd": "CADD_score",
    "Allele Frequency": "gnomAD_AF",
    "sift_max": "SIFT_score",
    "polyphen_max": "PolyPhen_score",
    "phylop": "Conservation_score",
    "Transcript Consequence": "Consequence"
})

consequence_map = {
    # Non-coding / lowest impact
    "5_prime_UTR_variant": 0,
    "3_prime_UTR_variant": 0,
    "intron_variant": 0,

    # Silent
    "synonymous_variant": 1,

    # Protein-altering (moderate)
    "missense_variant": 2,
    "inframe_insertion": 2,
    "inframe_deletion": 2,
    "protein_altering_variant": 2,

    # Splicing-related
    "splice_region_variant": 3,
    "splice_donor_variant": 4,
    "splice_acceptor_variant": 4,

    # Start/stop disruptions
    "start_lost": 4,
    "stop_retained_variant": 4,
    "stop_lost": 5,
    "stop_gained": 5,

    # Highest impact
    "frameshift_variant": 6
}
# Normalize text first (safety)
df["Consequence"] = (
    df["Consequence"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Map using lowercase keys
consequence_map = {k.lower(): v for k, v in consequence_map.items()}

df["Consequence_encoded"] = df["Consequence"].map(consequence_map)
# Assign lowest severity to unknowns instead of dropping rows
df["Consequence_encoded"] = df["Consequence_encoded"].fillna(0)

print(
    df["Consequence"]
    .value_counts()
)

print(
    df.groupby("Consequence")["Consequence_encoded"]
    .unique()
)
FINAL_FEATURES = [
    "CADD_score",
    "gnomAD_AF",
    "SIFT_score",
    "PolyPhen_score",
    "Consequence_encoded",
    "Conservation_score"
]

df = df[FINAL_FEATURES + ["Label"]]
# Create missingness indicators
df["SIFT_missing"] = df["SIFT_score"].isna().astype(int)
df["PolyPhen_missing"] = df["PolyPhen_score"].isna().astype(int)

# Replace NaNs with biologically neutral values
df["SIFT_score"] = df["SIFT_score"].fillna(1.0)
df["PolyPhen_score"] = df["PolyPhen_score"].fillna(0.0)

print(df.shape)
df.head()

id="imbalance_check"
print(df["Label"].value_counts())
print(df["Label"].value_counts(normalize=True))

# Features & label
X = df.drop("Label", axis=1)
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)

#With class imbalance
id="xgb_balance"
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42
)
xgb_model.fit(X_train, y_train)
print(X_train.columns.tolist())

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

def test_xgboost(model, X_test, y_test, threshold=0.5):
    """
    Evaluate XGBoost pathogenicity classifier
    """

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall (Sensitivity)": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    print("XGBoost Performance")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\n📄 Classification Report")
    print(classification_report(y_test, y_pred))

    return results
xgb_results = test_xgboost(xgb_model, X_test, y_test)

FEATURE_ORDER = [
    'CADD_score', 'gnomAD_AF', 'SIFT_score', 'PolyPhen_score', 'Consequence_encoded', 'Conservation_score', 'SIFT_missing', 'PolyPhen_missing'
]

def predict_variant_xgb(
    model,
    CADD_score,
    gnomAD_AF,
    SIFT_score,
    PolyPhen_score,
    Consequence_encoded,
    Conservation_score
):
    """
    Predict pathogenicity using XGBoost
    """

    # Handle missing SIFT / PolyPhen
    SIFT_missing = int(SIFT_score is None)
    PolyPhen_missing = int(PolyPhen_score is None)

    SIFT_score = 1.0 if SIFT_score is None else SIFT_score
    PolyPhen_score = 0.0 if PolyPhen_score is None else PolyPhen_score

    # Create input dataframe
    X_input = pd.DataFrame([[
        CADD_score,
        gnomAD_AF,
        SIFT_score,
        PolyPhen_score,
        SIFT_missing,
        PolyPhen_missing,
        Consequence_encoded,
        Conservation_score
    ]], columns=FEATURE_ORDER)

    # Predict probability
    prob = model.predict_proba(X_input)[0][1]

    label = int(prob >= 0.5)

    interpretation = (
        "Pathogenic / Likely Pathogenic"
        if label == 1
        else "Benign / Likely Benign"
    )

    return {
        "prediction": label,
        "probability": round(prob, 4),
        "interpretation": interpretation
    }

#Example usage
result = predict_variant_xgb(
    model=xgb_model,
    CADD_score=8.82,
    gnomAD_AF=0.000001,
    SIFT_score=0,
    PolyPhen_score=0,
    Consequence_encoded=1,   # missense
    Conservation_score=8.85
)

print(result)