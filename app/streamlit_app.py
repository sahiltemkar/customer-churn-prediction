
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_tuned.joblib"
GLOBAL_SHAP_PATH = ROOT / "reports" / "shap_feature_importance.csv"

THRESHOLD = 0.45

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model_and_explainer():
    pipeline = joblib.load(MODEL_PATH)
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    return pipeline, explainer

model, explainer = load_model_and_explainer()

st.title("Customer Churn Intelligence")
st.write(
    "Explore customer churn risk, understand the model's reasoning, "
    "and support data-informed retention decisions."
)

st.info(
    "This is a portfolio demonstration trained on the IBM Telco Customer "
    "Churn dataset. Predictions are estimates, not guarantees."
)

with st.form("customer_form"):
    st.subheader("Customer profile")
    st.caption("Enter the customer's details, then select Predict churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input(
            "Monthly charges", min_value=0.0, max_value=200.0,
            value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total charges", min_value=0.0, max_value=10000.0,
            value=840.0, step=10.0
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        paperless_billing = st.selectbox(
            "Paperless billing", ["Yes", "No"]
        )

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )

    with col3:
        online_security = st.selectbox(
            "Online security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["Yes", "No", "No internet service"]
        )

    submitted = st.form_submit_button(
        "Predict churn", type="primary", width="stretch"
    )

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    probability = float(model.predict_proba(customer)[0, 1])
    prediction = probability >= THRESHOLD

    st.divider()
    st.subheader("Prediction result")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Predicted churn probability", f"{probability:.1%}")
    metric2.metric("Decision threshold", f"{THRESHOLD:.0%}")
    metric3.metric(
        "Predicted class",
        "Churn risk" if prediction else "Lower churn risk"
    )

    st.progress(probability)

    if prediction:
        st.warning(
            "The predicted probability meets or exceeds the selected "
            "threshold. Consider reviewing this customer's needs."
        )
    else:
        st.success(
            "The predicted probability is below the selected threshold. "
            "This does not guarantee the customer will stay."
        )

    st.caption(
        "The threshold was selected to maximize F1 among the tested "
        "training out-of-fold thresholds. It has not been optimized for "
        "retention costs or profit."
    )

    # Explain this customer's prediction using the fitted preprocessing
    # pipeline and the Random Forest classifier.
    st.subheader("Why did the model make this prediction?")
    st.write(
        "SHAP values show which transformed features push this model's "
        "output toward or away from churn for this customer. They describe "
        "model behavior, not causation."
    )

    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(customer)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
    )

    shap_values = explainer.shap_values(transformed_df)

    # Handle common SHAP output formats for binary classifiers.
    if isinstance(shap_values, list):
        customer_shap = shap_values[1][0]
    else:
        values = getattr(shap_values, "values", shap_values)
        if getattr(values, "ndim", 0) == 3:
            customer_shap = values[0, :, 1]
        else:
            customer_shap = values[0]

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Customer value": transformed_df.iloc[0].values,
        "SHAP value": customer_shap,
    })

    explanation["Direction"] = explanation["SHAP value"].apply(
        lambda value: "Toward churn" if value > 0 else "Away from churn"
    )

    explanation["Feature"] = (
        explanation["Feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("numeric__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    positive = explanation[explanation["SHAP value"] > 0].nlargest(
        5, "SHAP value"
    )
    negative = explanation[explanation["SHAP value"] < 0].nsmallest(
        5, "SHAP value"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Factors pushing toward churn**")
        if positive.empty:
            st.write("No positive contributions for this customer.")
        else:
            st.dataframe(
                positive[["Feature", "Customer value", "SHAP value"]],
                hide_index=True,
                width="stretch",
            )

    with right:
        st.markdown("**Factors pushing away from churn**")
        if negative.empty:
            st.write("No negative contributions for this customer.")
        else:
            st.dataframe(
                negative[["Feature", "Customer value", "SHAP value"]],
                hide_index=True,
                width="stretch",
            )

    st.caption(
        "SHAP values are contributions to the model output, not necessarily "
        "percentage-point changes in churn probability. Correlated features "
        "and one-hot encoding can affect how contributions are distributed."
    )

    with st.expander("View the submitted customer profile"):
        st.dataframe(customer, hide_index=True, width="stretch")

st.divider()
st.subheader("Global model insights")

if GLOBAL_SHAP_PATH.exists():
    global_importance = pd.read_csv(GLOBAL_SHAP_PATH).head(10).copy()
    global_importance["Feature"] = (
        global_importance["Feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("numeric__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )
    st.bar_chart(
        global_importance.set_index("Feature")["Mean Absolute SHAP"],
        width="stretch",
    )
    st.caption(
        "Global importance summarizes the average absolute SHAP values "
        "across the analyzed sample. It does not show the direction of an "
        "effect or explain an individual prediction."
    )
else:
    st.warning(
        "Global SHAP report not found. Run src/explain_model.py first."
    )

st.caption(
    "Built with pandas, scikit-learn, XGBoost, SHAP and Streamlit."
)
