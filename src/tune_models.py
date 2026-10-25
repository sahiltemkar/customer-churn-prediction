from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# 1. Define project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the cleaned dataset
df = pd.read_csv(DATA_PATH)

# Remove customerID because it identifies customers but does not
# provide a useful general pattern for predicting churn.
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected labels found in the Churn column.")


# 3. Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()


# 4. Build preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# 5. Split data before tuning to keep the test set untouched
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# 6. Define models and candidate hyperparameters
search_spaces = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"],
        },
    },

    "random_forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=1),
        "params": {
            "classifier__n_estimators": [150, 250, 400],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__max_features": ["sqrt", 0.8],
            "classifier__class_weight": [None, "balanced"],
        },
    },

    "xgboost": {
        "model": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        ),
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__scale_pos_weight": [1, 2],
        },
    },
}


# 7. Tune each model using 5-fold cross-validation
results = []

for name, config in search_spaces.items():
    print(f"\r\n{'=' * 60}")
    print(f"Tuning {name}...")
    print("This may take several minutes.")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", config["model"]),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config["params"],
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=1,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train)

    # The best estimator has already been refitted on all training data.
    best_model = search.best_estimator_

    # Evaluate only after tuning, on the untouched test set.
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Best CV ROC-AUC": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Test Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Test Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Test F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Test ROC-AUC": roc_auc_score(y_test, probabilities),
        "Best Parameters": str(search.best_params_),
    }

    results.append(metrics)

    model_path = MODEL_DIR / f"{name}_tuned.joblib"
    joblib.dump(best_model, model_path)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Test ROC-AUC: {metrics['Test ROC-AUC']:.4f}")
    print(f"Test F1: {metrics['Test F1']:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Saved model: {model_path}")


# 8. Save a comparison report
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Best CV ROC-AUC", ascending=False
)

report_path = REPORT_DIR / "tuned_model_comparison.csv"
results_df.to_csv(report_path, index=False)

print("\r\n" + "=" * 60)
print("TUNING COMPLETE")
print(results_df.to_string(index=False))
print(f"\r\nSaved comparison report: {report_path}")
