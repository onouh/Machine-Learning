import pandas as pd

df = pd.read_csv("data/raw/H2.csv")

print("Shape (rows, columns):", df.shape)
print()
print("Column names:")
for col in df.columns:
    print(" -", col)

print()
print("Data types:")
print(df.dtypes)

# Strip whitespace from all text (object/str) columns.
text_columns = df.select_dtypes(include="object").columns
for col in text_columns:
    df[col] = df[col].str.strip()

# Replace the literal "NULL" token with real missing values (NaN).
df[text_columns] = df[text_columns].replace("NULL", pd.NA)

print()
print("Missing values per column (after whitespace fix + NULL-token fix):")
missing_counts = df.isna().sum()
print(missing_counts[missing_counts > 0])

print()
num_duplicates = df.duplicated().sum()
print("Number of exact duplicate rows:", num_duplicates)

print()
print("Example: counts of each duplicate group (top 10 most repeated rows):")
duplicate_group_sizes = df.groupby(list(df.columns), dropna=False).size()
duplicate_group_sizes = duplicate_group_sizes[duplicate_group_sizes > 1]
print(duplicate_group_sizes.sort_values(ascending=False).head(10))

print()
print("Total number of distinct row-patterns that have duplicates:", len(duplicate_group_sizes))
print("Sanity check - total extra copies implied:", (duplicate_group_sizes - 1).sum())

print()
is_dup = df.duplicated(keep=False)
print("Total rows involved in any duplicate group:", is_dup.sum())
print()
print("MarketSegment distribution among duplicate-group rows:")
print(df.loc[is_dup, "MarketSegment"].value_counts(normalize=True))
print()
print("MarketSegment distribution among NON-duplicate rows:")
print(df.loc[~is_dup, "MarketSegment"].value_counts(normalize=True))
print()
print("IsCanceled rate among duplicate-group rows:", df.loc[is_dup, "IsCanceled"].mean())
print("IsCanceled rate among NON-duplicate rows:", df.loc[~is_dup, "IsCanceled"].mean())
print()
print("DepositType distribution among duplicate-group rows:")
print(df.loc[is_dup, "DepositType"].value_counts(normalize=True))

print()
print("ADR summary statistics:")
print(df["ADR"].describe())
print()
print("ADR skewness:", df["ADR"].skew())
print()
print("Number of ADR values <= 0:", (df["ADR"] <= 0).sum())
print("Number of ADR values < 0 (negative):", (df["ADR"] < 0).sum())
print("Number of ADR values == 0:", (df["ADR"] == 0).sum())
print()
print("Top 10 highest ADR values:")
print(df["ADR"].sort_values(ascending=False).head(10))

print()
print("Full details of the row with ADR = 5400:")
print(df.loc[8455])

print()
print("ADR distribution for Room Type A bookings only:")
room_a = df[df["ReservedRoomType"] == "A"]
print(room_a["ADR"].describe())

print()
print("How many Room Type A bookings have ADR > 500?")
print((room_a["ADR"] > 500).sum())

print()
print("Full details of the row with ADR = 510:")
print(df.loc[71343])

print()
print("ADR distribution for AssignedRoomType G bookings only:")
room_g = df[df["AssignedRoomType"] == "G"]
print(room_g["ADR"].describe())

print()
print("Checking remaining top ADR outliers against their AssignedRoomType peer group:")
suspicious_indexes = [63852, 63729, 78455, 53786, 32906, 63713, 56938, 75481]
for idx in suspicious_indexes:
    row = df.loc[idx]
    peer_group = df[df["AssignedRoomType"] == row["AssignedRoomType"]]
    print(f"Index {idx}: ADR={row['ADR']}, AssignedRoomType={row['AssignedRoomType']}, "
          f"peer median={peer_group['ADR'].median():.2f}, peer 75th%={peer_group['ADR'].quantile(0.75):.2f}, "
          f"peer max={peer_group['ADR'].max():.2f}, ReservationStatus={row['ReservationStatus']}")

print()
print("Investigating ADR = 0 rows:")
zero_adr = df[df["ADR"] == 0]
print("Count:", len(zero_adr))
print()
print("MarketSegment breakdown (zero-ADR rows):")
print(zero_adr["MarketSegment"].value_counts())
print()
print("DepositType breakdown (zero-ADR rows):")
print(zero_adr["DepositType"].value_counts())
print()
print("IsCanceled rate (zero-ADR rows):", zero_adr["IsCanceled"].mean())
print("IsCanceled rate (overall dataset):", df["IsCanceled"].mean())
print()
print("CustomerType breakdown (zero-ADR rows):")
print(zero_adr["CustomerType"].value_counts())

print()
non_comp_zero = zero_adr[zero_adr["MarketSegment"] != "Complementary"]
print("Non-complementary zero-ADR rows count:", len(non_comp_zero))
print()
print("CustomerType breakdown (non-complementary zero-ADR rows):")
print(non_comp_zero["CustomerType"].value_counts())
print()
print("Stay length (nights) summary for non-complementary zero-ADR rows:")
stay_length = non_comp_zero["StaysInWeekendNights"] + non_comp_zero["StaysInWeekNights"]
print(stay_length.describe())
print()
print("How many non-complementary zero-ADR rows have 0 total nights?")
print((stay_length == 0).sum())
print()
print("ArrivalDateYear breakdown (non-complementary zero-ADR rows):")
print(non_comp_zero["ArrivalDateYear"].value_counts())
print()
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("Sample of 10 non-complementary zero-ADR rows:")
print(non_comp_zero[["ArrivalDateYear", "MarketSegment", "DistributionChannel", "Agent", "Company", "CustomerType", "StaysInWeekendNights", "StaysInWeekNights", "ReservationStatus"]].head(10))
print()
zero_nights = non_comp_zero[(non_comp_zero["StaysInWeekendNights"] + non_comp_zero["StaysInWeekNights"]) == 0]
print("ReservationStatus breakdown (0 nights, non-complementary, ADR=0):")
print(zero_nights["ReservationStatus"].value_counts())
print()
print("MarketSegment breakdown (0 nights, non-complementary, ADR=0):")
print(zero_nights["MarketSegment"].value_counts())

print()
print("=== Null check: how much, and does ADR differ when missing? ===")
for col in ["Children", "Country", "Agent", "Company"]:
    n_missing = df[col].isna().sum()
    print(f"{col}: {n_missing} missing ({n_missing / len(df) * 100:.2f}%), "
          f"distinct non-missing values: {df[col].nunique()}")

print()
for col in ["Agent", "Company"]:
    present = df[col].notna()
    print(f"{col}: median ADR when present = {df.loc[present, 'ADR'].median():.2f}, "
          f"when missing = {df.loc[~present, 'ADR'].median():.2f}")
    print(f"{col}: mean ADR when present = {df.loc[present, 'ADR'].mean():.2f}, "
          f"when missing = {df.loc[~present, 'ADR'].mean():.2f}")
    