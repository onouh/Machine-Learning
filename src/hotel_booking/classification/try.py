import os
import numpy as np
import pandas as pd
#new csv file for the cleaned data
# ---------------- 1. LOAD ----------------
df = pd.read_csv(r"data\processed\H2_noduplicates.csv")
df.columns = df.columns.str.strip()

# strip spaces inside text values, and turn the text "NULL" into real NaN
for c in df.select_dtypes(exclude="number").columns:
    df[c] = df[c].astype(str).str.strip()
df = df.replace("NULL", np.nan)
print("Raw shape:", df.shape)

# ---------------- 2. DROP LEAKAGE ----------------
df = df.drop(columns=["ReservationStatus", "ReservationStatusDate"])

# ---------------- 3. MISSING VALUES ----------------
df["Children"] = df["Children"].fillna(0)
df["Country"] = df["Country"].fillna("Unknown")

df["has_agent"] = df["Agent"].notna().astype(int)   # must run AFTER NULL -> NaN
df = df.drop(columns=["Agent", "Company"])

# ---------------- 4. INVALID ROWS ----------------
df["total_guests"] = df["Adults"] + df["Children"] + df["Babies"]
df["total_nights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]

df = df[df["total_guests"] > 0]
df = df[df["total_nights"] > 0]
df = df[df["Babies"] <= 5]
#df = df[df["ADR"] < 1000]

# ---------------- 5. ONE DATE FEATURE ----------------
months = {m: i for i, m in enumerate(
    ["January","February","March","April","May","June","July",
     "August","September","October","November","December"], 1)}

df["arrival_date"] = pd.to_datetime(dict(
    year=df["ArrivalDateYear"],
    month=df["ArrivalDateMonth"].map(months),
    day=df["ArrivalDateDayOfMonth"]))
df["arrival_days"] = (df["arrival_date"] - df["arrival_date"].min()).dt.days

df = df.drop(columns=["ArrivalDateYear", "ArrivalDateMonth",
                      "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth"])

# ---------------- 6. OTHER FEATURES ----------------
df["has_children"] = ((df["Children"] + df["Babies"]) > 0).astype(int)
df["adr_per_person"] = df["ADR"] / df["total_guests"]
df["is_prev_canceller"] = (df["PreviousCancellations"] > 0).astype(int)

keep = df["Country"].value_counts().nlargest(15).index
df["Country"] = df["Country"].where(df["Country"].isin(keep), "Other")

# ---------------- 7. SAVE THE NEW CSV ----------------
os.makedirs(r"data\processed", exist_ok=True)
out_path = r"data\processed\H2_clean.csv"
df.to_csv(out_path, index=False)
print("Saved:", out_path, df.shape)
print(df.isnull().sum().sum(), "missing values left")
print(df.columns.tolist())