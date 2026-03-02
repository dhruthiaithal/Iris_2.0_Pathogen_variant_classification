from google.colab import files
uploaded = files.upload() #upload csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

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

X = df.drop("Label", axis=1)
y = df["Label"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

nn_model.summary()

#Class imbalance
id="nn_balance"
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict
)
nn_eval = nn_model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"NN Accuracy: {nn_eval[1]:.4f}")
print(f"NN ROC-AUC: {nn_eval[2]:.4f}")
def test_neural_network(model, X_test, y_test, threshold=0.5):
    """
    Evaluate Neural Network pathogenicity classifier
    """

    # Predict probabilities
    y_prob = model.predict(X_test).ravel()

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

    print("Neural Network Performance")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return results
nn_results = test_neural_network(nn_model, X_test_scaled, y_test)

FEATURE_ORDER = [
    'CADD_score', 'gnomAD_AF', 'SIFT_score', 'PolyPhen_score', 'Consequence_encoded', 'Conservation_score', 'SIFT_missing', 'PolyPhen_missing'
]