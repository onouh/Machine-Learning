"""Cancellation classification experiments and inference pipeline."""

import pandas as pd
df = pd.read_csv(r"data\processed\H2_noduplicates.csv")
# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# 3. Remove the two data leakage columns
df = df.drop(columns=[
    "ReservationStatus",
    "ReservationStatusDate"
])

# 4. Separate the input features (X)
X = df.drop(columns=["IsCanceled"])
y = df["IsCanceled"]
print("\nRemaining features:")
print(X.columns.tolist())

df["Children"] = df["Children"].fillna(0)
df["Country"] = df["Country"].fillna("Unknown")

df["has_agent"] = df["Agent"].notna().astype(int)   # 1 = booked through an agent
df = df.drop(columns=["Company"])          # drop both original columns
#df = df.drop(columns=["Agent"]) 
# ---------------- 4. INVALID ROWS ----------------
df["total_guests"] = df["Adults"] + df["Children"] + df["Babies"]
df["total_nights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]

df = df[df["total_guests"] > 0]
df = df[df["total_nights"] > 0]

df = df[df["Babies"] <= 5]

df["has_children"] = ((df["Children"] + df["Babies"]) > 0).astype(int)

                   