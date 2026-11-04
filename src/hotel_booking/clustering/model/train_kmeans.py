import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = ROOT / "data" / "processed" / "H2_cleaned.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "H2_clustered.csv"
PREDICTION_PATH = ROOT / "data" / "processed" / "H2_prediction_10pct.csv"
MODEL_PATH = ROOT / "kmeans_model.pkl"


# 1. Load dataset

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Select features

numeric_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "ADR",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "TotalOfSpecialRequests"
]

categorical_features = [
    "Meal",
    "MarketSegment",
    "CustomerType",
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

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 5. Create K-Means model

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("kmeans", KMeans(
        n_clusters=3,
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


# 9. Predict clusters for the complete dataset

df["Cluster"] = model.predict(df[all_features])

print("\nNumber of bookings in each cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Profiles:")
print(df.groupby("Cluster")[numeric_features].mean().round(2))


# 10. Save model and outputs

joblib.dump(model, MODEL_PATH)
print("\nTrained model saved at:", MODEL_PATH)

df.to_csv(OUTPUT_PATH, index=False)
print("Clustered dataset saved at:", OUTPUT_PATH)

print("\nK-Means training process completed successfully!")