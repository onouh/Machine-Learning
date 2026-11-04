import pandas as pd
import joblib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = ROOT / "kmeans_model.pkl"

# The 10% prediction set saved by train_kmeans.py (never seen in training/validation)
NEW_DATA_PATH = ROOT / "data" / "processed" / "H2_prediction_10pct.csv"

OUTPUT_PATH = ROOT / "data" / "new_bookings_clustered.csv"


# 1. Features: must match train_kmeans.py (CAT_MODE = "none", USE_ADR = False)

numeric_features = [
    "LeadTime",
    "TotalNights",
    "TotalGuests",
    "TotalOfSpecialRequests"
]

categorical_features = []

all_features = numeric_features + categorical_features


# 2. Load model and new data

model = joblib.load(MODEL_PATH)
print("Trained model loaded successfully!")

new_data = pd.read_csv(NEW_DATA_PATH)
print("New data loaded:", new_data.shape)


# 3. Same feature engineering as training (row-wise, no leakage)

raw_required = [
    "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "TotalOfSpecialRequests"
]

missing_raw = [col for col in raw_required if col not in new_data.columns]
if missing_raw:
    raise ValueError(f"Missing columns: {missing_raw}")

new_data["TotalNights"] = new_data["StaysInWeekendNights"] + new_data["StaysInWeekNights"]
new_data["TotalGuests"] = new_data["Adults"] + new_data["Children"].fillna(0) + new_data["Babies"]


# 4. Predict clusters

X_new = new_data[all_features].copy()

new_data["Cluster"] = model.predict(X_new)

print("\nPredicted cluster labels:")
print(new_data["Cluster"].value_counts().sort_index())

print("\nCluster share (%):")
print((new_data["Cluster"].value_counts(normalize=True).sort_index() * 100).round(2))

print("\nCluster Profiles:")
print(new_data.groupby("Cluster")[numeric_features].mean().round(2))


# 5. Save results

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
new_data.to_csv(OUTPUT_PATH, index=False)

print("\nNew bookings clustered successfully!")
print("Results saved at:", OUTPUT_PATH)