
# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using the IBM Telco Customer Churn dataset. The project compares Logistic Regression, Random Forest, and XGBoost, uses SHAP for model explainability, and provides an interactive Streamlit application.

## Project Overview

Customer churn occurs when customers stop using a company's services. Identifying customers who may churn can help businesses investigate retention opportunities and prioritize customer outreach.

This project covers data cleaning, exploratory data analysis, model training, hyperparameter tuning, threshold selection, model explainability, and interactive predictions.

## Features

- Data cleaning and validation
- Exploratory data analysis with visualizations
- Three classification algorithms:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Five-fold cross-validation and randomized hyperparameter search
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Threshold selection using out-of-fold predictions
- Global and individual-customer SHAP explanations
- Streamlit application for interactive churn predictions

## Dataset

The project uses the IBM Telco Customer Churn dataset, containing 7,043 customer records and 21 original columns.

The target variable is `Churn`, which indicates whether a customer left the service.

The cleaning script converts `TotalCharges` to numeric and handles 11 blank values associated with zero-tenure customers.

## Model Evaluation

The following results are from the held-out test set. The tuned model metrics use the default classification threshold of 0.50 unless stated otherwise.

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 80.55% | 65.72% | 55.88% | 0.6040 | 0.8420 |
| Random Forest (baseline) | 78.28% | 61.64% | 48.13% | 0.5405 | 0.8196 |
| XGBoost (baseline) | 80.20% | 65.99% | 52.41% | 0.5842 | 0.8442 |
| Logistic Regression (tuned) | 80.48% | 65.52% | 55.88% | 0.6032 | 0.8412 |
| Random Forest (tuned) | 76.79% | 54.71% | 72.99% | 0.6254 | 0.8418 |
| XGBoost (tuned) | 80.48% | 66.89% | 52.41% | 0.5877 | 0.8476 |

The tuned XGBoost model achieved the highest test ROC-AUC among these runs. The tuned Random Forest achieved the highest F1-score at the default threshold and higher churn recall.

## Threshold Selection

The tuned Random Forest's classification threshold was selected by maximizing F1 across tested thresholds using training out-of-fold predictions.

At a threshold of 0.45, the held-out test results were:

- Accuracy: 75.94%
- Precision: 53.15%
- Recall: 78.88%
- F1-score: 0.6351
- ROC-AUC: 0.8418

The confusion matrix was:

| | Predicted stay | Predicted churn |
|---|---:|---:|
| Actual stay | 775 | 260 |
| Actual churn | 79 | 295 |

This threshold catches more churners but also produces more false positives. It has not been optimized for financial outcomes or retention costs.

## Model Explainability

SHAP is used to understand the influence of features on model predictions.

The global analysis ranked month-to-month contract type, tenure, fiber-optic internet service, total charges, and lack of online security among the most influential features in the analyzed sample.

The Streamlit app also displays individual-customer SHAP contributions. These describe the model's behavior, not causation.

## Project Structure

```text
Customer Churn Prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── clean_data.py
│   ├── eda.py
│   ├── train_models.py
│   ├── tune_models.py
│   ├── threshold_tuning.py
│   └── explain_model.py
├── check_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Python 3.10 or a compatible Python version is recommended.

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Run the Streamlit application.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

The application should open at `http://localhost:8501`.

The app loads the saved model from `models/random_forest_tuned.joblib`. Ensure the trained model file is included in the repository.

## Limitations

- The model is trained on a specific telco dataset and may not generalize to other companies or time periods.
- Model-predicted probabilities are not guaranteed to be calibrated real-world churn probabilities.
- Threshold choice depends on business costs and retention strategy.
- SHAP explanations describe learned associations and model behavior, not causal effects.
- Predictions should support human decisions rather than be treated as guarantees.

## Tools and Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Streamlit, and joblib.
