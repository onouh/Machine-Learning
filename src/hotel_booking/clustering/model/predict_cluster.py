import pandas as pd
import joblib
from pathlib import Path



ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = ROOT / "kmeans_model.pkl"

NEW_DATA_PATH = ROOT / "data" / "processed" / "H2_cleaned.csv"

OUTPUT_PATH = ROOT / "data" / "new_bookings_clustered.csv"





numeric_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
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




model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")




new_data = pd.read_csv(NEW_DATA_PATH)

print("New data loaded:", new_data.shape)




missing_columns = [
    col for col in all_features
    if col not in new_data.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")



X_new = new_data[all_features].copy()



predicted_labels = model.predict(X_new)




new_data["Cluster"] = predicted_labels




print("\nPredicted cluster labels:")

print(new_data[["Cluster"]].value_counts().sort_index())



new_data.to_csv(OUTPUT_PATH, index=False)

print("\nNew bookings clustered successfully!")

print("Results saved at:", OUTPUT_PATH)