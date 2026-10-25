from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "cleaned_churn.csv"
MODEL_PATH = ROOT / "models" / "random_forest_tuned.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the cleaned dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate the same split used during model training.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 2. Load the fitted Random Forest pipeline
model = joblib.load(MODEL_PATH)

# 3. Transform the test features using the fitted preprocessor
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_test_transformed = preprocessor.transform(X_test)

# Recover names for the transformed features, including one-hot encoded ones.
feature_names = preprocessor.get_feature_names_out()

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index,
)

# 4. Use a reproducible sample to keep SHAP calculations manageable.
X_explain = X_test_transformed.sample(
    n=min(300, len(X_test_transformed)),
    random_state=42,
)

# 5. Calculate SHAP values for the fitted Random Forest.
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_explain)

# SHAP output shape can differ across SHAP versions.
# Select the values for the churn class (class 1).
if isinstance(shap_values, list):
    churn_shap_values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    # Common shape: samples x features x classes
    churn_shap_values = shap_values[:, :, 1]
else:
    churn_shap_values = shap_values

# 6. Save a global feature importance plot.
plt.figure()
shap.summary_plot(
    churn_shap_values,
    X_explain,
    show=False,
    max_display=15,
)
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# 7. Save a bar chart of mean absolute SHAP values.
plt.figure()
shap.summary_plot(
    churn_shap_values,
    X_explain,
    plot_type="bar",
    show=False,
    max_display=15,
)
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# 8. Save a numeric feature-importance table.
importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean Absolute SHAP": abs(churn_shap_values).mean(axis=0),
}).sort_values("Mean Absolute SHAP", ascending=False)

importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False,
)

print("\r\nTop 15 features by mean absolute SHAP value:")
print(importance.head(15).to_string(index=False))

print("\r\nSaved:")
print("reports/shap_summary.png")
print("reports/shap_feature_importance.png")
print("reports/shap_feature_importance.csv")
