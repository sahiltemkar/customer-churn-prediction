from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the cleaned dataset
df = pd.read_csv(DATA_PATH)

# Customer ID is an identifier, not a predictive feature
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in the Churn target.")

# Explicit feature groups
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]


# 3. Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# 4. Split data before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("TRAINING DATA")
print("-" * 50)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training churn rate:", round(y_train.mean() * 100, 2), "%")


# 5. Define baseline models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}


# 6. Cross-validate, train and evaluate each model
results = []

for name, classifier in models.items():
    print(f"\r\n{'=' * 50}")
    print(f"MODEL: {name}")
    print("=" * 50)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Five-fold CV on training data only
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )

    print(
        f"5-fold CV ROC-AUC: "
        f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
    )

    # Fit using the full training split
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test data
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "CV ROC-AUC": cv_scores.mean(),
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Test ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)

    print("\r\nTest metrics:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\r\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No churn", "Churn"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Save the complete fitted pipeline
    model_filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, MODEL_DIR / model_filename)


# 7. Save a model comparison table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "CV ROC-AUC", ascending=False
)

results_path = REPORTS_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\r\nFINAL MODEL COMPARISON")
print("=" * 80)
print(results_df.round(4).to_string(index=False))
print("\r\nComparison saved to:", results_path)
print("Fitted pipelines saved in:", MODEL_DIR)
