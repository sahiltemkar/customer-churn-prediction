from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model_path = ROOT / "models" / "random_forest_tuned.joblib"
model = joblib.load(model_path)

# Generate out-of-fold probabilities using training data only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probabilities = cross_val_predict(
    clone(model),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

# Find the threshold with the highest out-of-fold F1.
thresholds = [i / 100 for i in range(20, 71, 5)]
threshold_results = []

for threshold in thresholds:
    predictions = (oof_probabilities >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_train, predictions, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df = threshold_df.sort_values("F1", ascending=False)

best_threshold = float(threshold_df.iloc[0]["Threshold"])

print("\r\nThreshold comparison (training out-of-fold predictions):")
print(threshold_df.to_string(index=False))
print(f"\r\nSelected threshold: {best_threshold:.2f}")

# Evaluate the selected threshold on the untouched test set.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print("\r\nTest-set results at selected threshold:")
print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
print(f"Precision: {precision_score(y_test, test_predictions):.4f}")
print(f"Recall:    {recall_score(y_test, test_predictions):.4f}")
print(f"F1:        {f1_score(y_test, test_predictions):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probabilities):.4f}")

print("\r\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\r\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0,
))

threshold_df.to_csv(
    ROOT / "reports" / "threshold_comparison.csv",
    index=False,
)
print("\r\nSaved reports/threshold_comparison.csv")
