
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_churn.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_PATH)

# Consistent chart styling
sns.set_theme(style="whitegrid")

print("Dataset shape:", df.shape)
print("\nOverall churn rate:")
print(f"{(df['Churn'] == 'Yes').mean() * 100:.2f}%")

# 1. Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "churn_distribution.png", dpi=150)
plt.close()

# 2. Churn rate by contract type
contract_churn = (
    df.groupby("Contract", observed=True)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nChurn rate by contract type (%):")
print(contract_churn.round(2))

plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="Contract",
    y=df["Churn"].eq("Yes").astype(int),
    errorbar=None
)
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "churn_by_contract.png", dpi=150)
plt.close()

# 3. Tenure distribution by churn outcome
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure by Churn Outcome")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "tenure_by_churn.png", dpi=150)
plt.close()

# 4. Monthly charges by churn outcome
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Outcome")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "monthly_charges_by_churn.png", dpi=150)
plt.close()

# Summary statistics
print("\nTenure summary by churn:")
print(df.groupby("Churn")["tenure"].median())

print("\nMonthly charges summary by churn:")
print(df.groupby("Churn")["MonthlyCharges"].median())

print("\nEDA completed. Charts saved in:", REPORTS_DIR)
