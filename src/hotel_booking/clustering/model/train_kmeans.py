import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "H2_deduplicated.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "H2_clustered.csv"
MODEL_PATH = ROOT / "kmeans_model.pkl"




df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)




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


missing_columns = [
    col for col in all_features if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

X_train = df[all_features].copy()




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




model = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("kmeans", KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ))
])




print("\nTraining K-Means model...")

model.fit(X_train)

print("Training completed!")




train_labels = model.predict(X_train)

df["Cluster"] = train_labels




kmeans = model.named_steps["kmeans"]

print("\nModel Evaluation")

print("Inertia:", kmeans.inertia_)

X_processed = model.named_steps["preprocessor"].transform(X_train)

silhouette = silhouette_score(
    X_processed,
    train_labels,
    sample_size=3000,
    random_state=42
)

print("Silhouette Score:", silhouette)



print("\nNumber of bookings in each cluster:")

print(df["Cluster"].value_counts().sort_index())




print("\nCluster Profiles:")

print(
    df.groupby("Cluster")[numeric_features].mean().round(2)
)




joblib.dump(model, MODEL_PATH)

print("\nTrained model saved at:", MODEL_PATH)




df.to_csv(OUTPUT_PATH, index=False)

print("Clustered dataset saved at:", OUTPUT_PATH)

print("\nK-Means training process completed successfully!")