from pathlib import Path
import pandas as pd

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "cleaned_churn.csv"

def clean_data():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    # Remove accidental leading/trailing spaces from text columns
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    # Convert TotalCharges to a numeric column
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Investigate missing TotalCharges
    missing_charges = df["TotalCharges"].isna()
    zero_tenure = df["tenure"].eq(0)

    # Only fill missing charges when tenure is zero
    valid_to_fill = missing_charges & zero_tenure
    df.loc[valid_to_fill, "TotalCharges"] = 0

    # Stop if any unexplained missing values remain
    if df.isna().any().any():
        missing_summary = df.isna().sum()
        missing_summary = missing_summary[missing_summary > 0]

        raise ValueError(
            "Unexpected missing values found:\r\n"
            f"{missing_summary}"
        )

    # Validate target values
    expected_targets = {"Yes", "No"}
    actual_targets = set(df["Churn"].unique())

    if actual_targets != expected_targets:
        raise ValueError(
            f"Unexpected Churn values: {actual_targets}"
        )

    # Save cleaned dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\r\nData cleaning completed!")
    print("Cleaned shape:", df.shape)
    print("TotalCharges data type:", df["TotalCharges"].dtype)
    print("Missing values remaining:", df.isna().sum().sum())
    print("Rows with zero tenure:", zero_tenure.sum())
    print("Missing charges filled:", valid_to_fill.sum())
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    clean_data()
