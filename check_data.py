
from pathlib import Path
import pandas as pd

data_path = Path("data/raw/Telco-Customer-Churn.csv")
df = pd.read_csv(data_path)

print("DATA QUALITY CHECK")
print("-" * 40)

# 1. Check duplicate records
print("Duplicate rows:", df.duplicated().sum())

# 2. Check unique customer IDs
print("Unique customer IDs:", df["customerID"].nunique())

# 3. Inspect TotalCharges
print("\nTotalCharges before conversion:")
print("Data type:", df["TotalCharges"].dtype)

blank_count = (
    df["TotalCharges"].astype(str).str.strip().eq("").sum()
)
print("Blank or whitespace entries:", blank_count)

total_charges = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("Missing after numeric conversion:", total_charges.isna().sum())

print("\nRows with missing TotalCharges:")
print(
    df.loc[
        total_charges.isna(),
        ["customerID", "tenure", "TotalCharges", "Churn"]
    ]
)

# 4. Inspect churn distribution
print("\nCHURN DISTRIBUTION")
print("-" * 40)
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# 5. Inspect numeric features
print("\nNUMERIC FEATURE SUMMARY")
print("-" * 40)
print(df[["tenure", "MonthlyCharges"]].describe())
