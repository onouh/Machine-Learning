import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = ROOT / "data" / "processed" / "H2_cleaned.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "H2_clustered.csv"
PREDICTION_PATH = ROOT / "data" / "processed" / "H2_prediction_10pct.csv"
MODEL_PATH = ROOT / "kmeans_model.pkl"
LOG_PATH = ROOT / "logs" / "kmeans_experiments.csv"

# Experiment settings: change these for each attempt
N_CLUSTERS =  3  # python train_kmeans.py 4
USE_ADR = False  # True = add ADR, clipped at the 99th percentile of the training set
USE_LOG = True   # True = apply log1p to the skewed numeric features
NOTES = (
    "no previous-booking flags in training; "
    + ("ADR clipped p99 included; " if USE_ADR else "ADR excluded; ")
    + ("log1p on skewed features" if USE_LOG else "no log transform")
)


# 1. Load dataset

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Select features

# Feature engineering: row-wise only (no statistics learned from the data),
# so doing it before the split does not cause leakage
raw_required = [
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children",
    "Babies", "PreviousCancellations", "PreviousBookingsNotCanceled"
]
missing_raw = [col for col in raw_required if col not in df.columns]
if missing_raw:
    raise ValueError(f"Missing columns: {missing_raw}")

df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
df["TotalGuests"] = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
df["HasPreviousCancellation"] = (df["PreviousCancellations"] > 0).astype(int)
df["HasPreviousBookings"] = (df["PreviousBookingsNotCanceled"] > 0).astype(int)

# The previous-booking flags are NOT training features: after scaling, a rare
# 0/1 flag dominates the distances. They are only used for profiling below.
numeric_features = [
    "LeadTime",
    "TotalNights",
    "TotalGuests",
    "TotalOfSpecialRequests"
]

if USE_ADR:
    df["ADR_clipped"] = df["ADR"]  # clipped after the split (see section 3)
    numeric_features.append("ADR_clipped")

# Skewed, non-negative features that get log1p before scaling
log_candidates = ["LeadTime", "TotalNights", "TotalOfSpecialRequests", "ADR_clipped"]
log_features = [c for c in numeric_features if USE_LOG and c in log_candidates]
linear_features = [c for c in numeric_features if c not in log_features]

categorical_features = [
    "Meal",
    "MarketSegment",
    "DepositType",
    "ReservedRoomType"
]

all_features = numeric_features + categorical_features

missing_columns = [col for col in all_features if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")


# 3. Split dataset: 70% training, 20% validation, 10% prediction

# Step 1: take 10% out as the prediction set (never used in training/validation)
df_rest, df_predict = train_test_split(
    df,
    test_size=0.10,
    random_state=42
)

# Step 2: split the remaining 90% into 70% / 20% of the original data
# 0.20 / 0.90 = 0.2222...
df_train, df_val = train_test_split(
    df_rest,
    test_size=0.20 / 0.90,
    random_state=42
)

# Clip ADR using a cap learned from the training set only (no leakage)
if USE_ADR:
    adr_cap = df_train["ADR"].quantile(0.99)
    for part in (df, df_train, df_val, df_predict):
        part["ADR_clipped"] = part["ADR"].clip(lower=0, upper=adr_cap)
    print("ADR cap (99th percentile of training):", adr_cap)

X_train = df_train[all_features].copy()
X_val = df_val[all_features].copy()

print("\nDataset split completed!")
print("Training data  (70%):", X_train.shape)
print("Validation data (20%):", X_val.shape)
print("Prediction data (10%):", df_predict.shape)

# Save the 10% prediction set to use in other files
df_predict.to_csv(PREDICTION_PATH, index=False)
print("Prediction set saved at:", PREDICTION_PATH)


# 4. Preprocessing pipelines

linear_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

log_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_linear", linear_pipeline, linear_features),
    ("num_log", log_pipeline, log_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 5. Create K-Means model

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("kmeans", KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    ))
])


# 6. Train model using 70%

print("\nTraining K-Means model...")

model.fit(X_train)

print("Training completed!")


# 7. Predict clusters for training and validation data

train_labels = model.predict(X_train)
val_labels = model.predict(X_val)


# 8. Evaluate model

kmeans = model.named_steps["kmeans"]
fitted_preprocessor = model.named_steps["preprocessor"]

print("\nModel Evaluation")
print("Training Inertia:", kmeans.inertia_)

X_train_processed = fitted_preprocessor.transform(X_train)
X_val_processed = fitted_preprocessor.transform(X_val)

# Validation inertia (sum of squared distances to nearest centroid)
val_inertia = -kmeans.score(X_val_processed)
print("Validation Inertia:", val_inertia)

# Silhouette scores
train_silhouette = None
val_silhouette = None

if 2 <= len(set(train_labels)) < len(train_labels):
    train_silhouette = silhouette_score(
        X_train_processed,
        train_labels,
        sample_size=min(3000, len(train_labels)),
        random_state=42
    )
    print("Training Silhouette Score:", train_silhouette)

if 2 <= len(set(val_labels)) < len(val_labels):
    val_silhouette = silhouette_score(
        X_val_processed,
        val_labels,
        sample_size=min(3000, len(val_labels)),
        random_state=42
    )
    print("Validation Silhouette Score:", val_silhouette)
else:
    print("Validation Silhouette Score cannot be calculated.")

# Davies-Bouldin (lower is better); needs dense input
X_val_dense = (
    X_val_processed.toarray()
    if hasattr(X_val_processed, "toarray")
    else X_val_processed
)
val_db = None
if 2 <= len(set(val_labels)) < len(val_labels):
    val_db = davies_bouldin_score(X_val_dense, val_labels)
    print("Validation Davies-Bouldin Score:", val_db)


# 9. Predict clusters for the complete dataset

df["Cluster"] = model.predict(df[all_features])

print("\nNumber of bookings in each cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Profiles:")
print(df.groupby("Cluster")[numeric_features].mean().round(2))

# Profile-only columns (not used in training unless USE_ADR is True)
profile_only = [
    c for c in ["ADR", "HasPreviousCancellation", "HasPreviousBookings"]
    if c in df.columns
]
print("\nProfile-only columns per cluster:")
print(df.groupby("Cluster")[profile_only].mean().round(2))

# Cluster share: check that no cluster is tiny (an inflated silhouette warning)
print("\nCluster share (%):")
print((df["Cluster"].value_counts(normalize=True).sort_index() * 100).round(2))


# 10. Save model and outputs

joblib.dump(model, MODEL_PATH)
print("\nTrained model saved at:", MODEL_PATH)

df.to_csv(OUTPUT_PATH, index=False)
print("Clustered dataset saved at:", OUTPUT_PATH)

# 11. Log this attempt (append only, never overwrites previous attempts)

cluster_sizes = df["Cluster"].value_counts().sort_index().to_dict()

log_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "k": N_CLUSTERS,
    "train_size": len(X_train),
    "val_size": len(X_val),
    "predict_size": len(df_predict),
    "train_inertia": round(kmeans.inertia_, 2),
    "val_inertia": round(val_inertia, 2),
    "train_inertia_per_sample": round(kmeans.inertia_ / len(X_train), 4),
    "val_inertia_per_sample": round(val_inertia / len(X_val), 4),
    "train_silhouette": None if train_silhouette is None else round(train_silhouette, 4),
    "val_silhouette": None if val_silhouette is None else round(val_silhouette, 4),
    "val_davies_bouldin": None if val_db is None else round(val_db, 4),
    "cluster_sizes": str(cluster_sizes),
    "notes": NOTES,
}

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame([log_row]).to_csv(
    LOG_PATH,
    mode="a",
    header=not LOG_PATH.exists(),
    index=False
)

print("Attempt logged at:", LOG_PATH)

print("\nK-Means training process completed successfully!")