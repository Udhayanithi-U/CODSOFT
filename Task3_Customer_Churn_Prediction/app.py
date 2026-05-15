"""Streamlit dashboard for customer churn prediction."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import prepare_prediction_data  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "Churn_Modelling.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.joblib"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_frame(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    features = prepare_prediction_data(raw_df)
    output = raw_df.copy()
    output["churn_probability"] = model.predict_proba(features)[:, 1]
    output["predicted_churn"] = model.predict(features)
    return output


def manual_customer_form() -> pd.DataFrame:
    left, middle, right = st.columns(3)

    with left:
        credit_score = st.slider("Credit score", 300, 900, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    with middle:
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure", 0, 10, 4)
        balance = st.number_input("Balance", min_value=0.0, value=65000.0, step=1000.0)

    with right:
        products = st.selectbox("Number of products", [1, 2, 3, 4])
        has_card = st.selectbox("Has credit card", [1, 0], format_func=lambda x: "Yes" if x else "No")
        active = st.selectbox("Active member", [1, 0], format_func=lambda x: "Yes" if x else "No")
        salary = st.number_input("Estimated salary", min_value=0.0, value=95000.0, step=1000.0)

    return pd.DataFrame(
        [
            {
                "RowNumber": 1,
                "CustomerId": 10000001,
                "Surname": "Demo",
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": products,
                "HasCrCard": has_card,
                "IsActiveMember": active,
                "EstimatedSalary": salary,
            }
        ]
    )


df = load_dataset()
comparison = load_comparison()
model_bundle = load_model()

st.title("Customer Churn Prediction")
st.caption("A simple dashboard for understanding and predicting customer churn")

if model_bundle is None:
    st.warning("Model not found. Run `python src/train_model.py` before using predictions.")
else:
    st.success(f"Loaded model: {model_bundle['model_name']}")

overview_tab, predict_tab, insights_tab = st.tabs(
    ["Overview", "Predict Churn", "Data Insights"]
)

with overview_tab:
    if df is not None:
        churn_rate = df["Exited"].mean() * 100
        active_rate = df["IsActiveMember"].mean() * 100

        metric_cols = st.columns(4)
        metric_cols[0].metric("Customers", f"{len(df):,}")
        metric_cols[1].metric("Churn Rate", f"{churn_rate:.1f}%")
        metric_cols[2].metric("Average Age", f"{df['Age'].mean():.1f}")
        metric_cols[3].metric("Active Members", f"{active_rate:.1f}%")

        st.subheader("Churn Split")
        churn_counts = df["Exited"].map({0: "Stayed", 1: "Churned"}).value_counts()
        st.bar_chart(churn_counts)
    else:
        st.info("Dataset not found. Extract `Churn_Modelling.csv` into the data folder.")

    if comparison is not None:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)
        st.bar_chart(
            comparison.set_index("model")[
                ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            ]
        )
    else:
        st.info("Model comparison report is not available yet. Train the model first.")

with predict_tab:
    st.subheader("Check One Customer")

    if model_bundle is not None:
        customer_df = manual_customer_form()

        if st.button("Predict Churn Risk", type="primary"):
            prediction = predict_frame(model_bundle["model"], customer_df)
            probability = float(prediction.loc[0, "churn_probability"])
            predicted_class = int(prediction.loc[0, "predicted_churn"])

            if predicted_class == 1:
                st.error(f"This customer has a high churn risk: {probability:.2%}")
            else:
                st.success(f"This customer is likely to stay. Churn risk: {probability:.2%}")

            st.dataframe(
                prediction[
                    [
                        "CreditScore",
                        "Geography",
                        "Gender",
                        "Age",
                        "Balance",
                        "NumOfProducts",
                        "IsActiveMember",
                        "churn_probability",
                        "predicted_churn",
                    ]
                ],
                use_container_width=True,
            )

        st.divider()
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            predictions = predict_frame(model_bundle["model"], uploaded_df)
            st.dataframe(predictions.head(100), use_container_width=True)
            st.download_button(
                "Download predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Train the model to enable predictions.")

with insights_tab:
    if df is not None:
        st.subheader("Churn by Geography")
        geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False) * 100
        st.bar_chart(geo_churn)

        st.subheader("Churn by Active Membership")
        active_churn = (
            df.assign(MemberStatus=df["IsActiveMember"].map({0: "Inactive", 1: "Active"}))
            .groupby("MemberStatus")["Exited"]
            .mean()
            .sort_values(ascending=False)
            * 100
        )
        st.bar_chart(active_churn)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info("Dataset not found.")
