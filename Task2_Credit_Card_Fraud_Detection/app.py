"""Streamlit dashboard for the Credit Card Fraud Detection project."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_model import add_features  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "best_fraud_model.joblib"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"


CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]

STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
)


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_model_comparison() -> pd.DataFrame | None:
    if not MODEL_COMPARISON_PATH.exists():
        return None
    return pd.read_csv(MODEL_COMPARISON_PATH)


def load_dataset_preview() -> pd.DataFrame | None:
    train_path = DATA_DIR / "fraudTrain.csv"
    if not train_path.exists():
        return None
    return pd.read_csv(train_path, nrows=20000)


def build_manual_transaction() -> pd.DataFrame:
    left, middle, right = st.columns(3)

    with left:
        amount = st.number_input("Transaction amount", min_value=0.0, value=120.0, step=10.0)
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index("shopping_net"))
        gender = st.selectbox("Gender", ["F", "M"])
        state = st.selectbox("State", STATES, index=STATES.index("CA"))

    with middle:
        transaction_date = st.date_input("Transaction date", value=date(2020, 6, 21))
        transaction_time = st.time_input("Transaction time", value=time(14, 30))
        dob = st.date_input("Customer date of birth", value=date(1985, 5, 15))
        city_population = st.number_input("City population", min_value=1, value=50000, step=1000)

    with right:
        customer_lat = st.number_input("Customer latitude", value=34.05)
        customer_long = st.number_input("Customer longitude", value=-118.24)
        merchant_lat = st.number_input("Merchant latitude", value=34.08)
        merchant_long = st.number_input("Merchant longitude", value=-118.25)

    transaction_dt = datetime.combine(transaction_date, transaction_time)

    return pd.DataFrame(
        [
            {
                "Unnamed: 0": 0,
                "trans_date_trans_time": transaction_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "cc_num": 0,
                "merchant": "demo_merchant",
                "category": category,
                "amt": amount,
                "first": "Demo",
                "last": "User",
                "gender": gender,
                "street": "Demo Street",
                "city": "Demo City",
                "state": state,
                "zip": 90001,
                "lat": customer_lat,
                "long": customer_long,
                "city_pop": city_population,
                "job": "Demo Job",
                "dob": dob.strftime("%Y-%m-%d"),
                "trans_num": "demo_transaction",
                "unix_time": int(transaction_dt.timestamp()),
                "merch_lat": merchant_lat,
                "merch_long": merchant_long,
            }
        ]
    )


def predict_dataframe(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    features = add_features(raw_df)
    if "is_fraud" in features.columns:
        features = features.drop(columns=["is_fraud"])

    output = raw_df.copy()
    output["fraud_probability"] = model.predict_proba(features)[:, 1]
    output["predicted_is_fraud"] = model.predict(features)
    return output


st.title("Credit Card Fraud Detection")
st.caption("Machine learning dashboard for detecting fraudulent transactions")

model_bundle = load_model()
comparison_df = load_model_comparison()
preview_df = load_dataset_preview()

if model_bundle is None:
    st.warning(
        "Trained model not found. Run `python src/train_model.py` first, then reopen this app."
    )
else:
    st.success(f"Loaded trained model: {model_bundle['model_name']}")

tab_overview, tab_predict, tab_data = st.tabs(
    ["Project Dashboard", "Fraud Prediction", "Dataset View"]
)

with tab_overview:
    metric_cols = st.columns(4)

    if preview_df is not None:
        fraud_rate = preview_df["is_fraud"].mean() * 100
        metric_cols[0].metric("Sample Rows Loaded", f"{len(preview_df):,}")
        metric_cols[1].metric("Fraud Rate", f"{fraud_rate:.2f}%")
        metric_cols[2].metric("Transaction Categories", preview_df["category"].nunique())
        metric_cols[3].metric("States Covered", preview_df["state"].nunique())
    else:
        st.info("Dataset preview unavailable. Place `fraudTrain.csv` inside the `data` folder.")

    if comparison_df is not None:
        st.subheader("Model Performance")
        st.dataframe(comparison_df, use_container_width=True)

        chart_data = comparison_df.set_index("model")[
            ["precision", "recall", "f1_score", "roc_auc", "average_precision"]
        ]
        st.bar_chart(chart_data)
    else:
        st.info("Model report not found yet. Train the model to generate comparison results.")

    if preview_df is not None:
        st.subheader("Fraud vs Legitimate Transactions")
        class_counts = preview_df["is_fraud"].map({0: "Legitimate", 1: "Fraud"}).value_counts()
        st.bar_chart(class_counts)

with tab_predict:
    st.subheader("Single Transaction Check")

    if model_bundle is not None:
        manual_df = build_manual_transaction()

        if st.button("Check Fraud Risk", type="primary"):
            result = predict_dataframe(model_bundle["model"], manual_df)
            probability = float(result.loc[0, "fraud_probability"])
            prediction = int(result.loc[0, "predicted_is_fraud"])

            if prediction == 1:
                st.error(f"High fraud risk detected. Probability: {probability:.2%}")
            else:
                st.success(f"Transaction looks legitimate. Fraud probability: {probability:.2%}")

            st.dataframe(
                result[["amt", "category", "gender", "state", "fraud_probability", "predicted_is_fraud"]],
                use_container_width=True,
            )

        st.divider()
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Upload a transaction CSV", type=["csv"])

        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            predictions = predict_dataframe(model_bundle["model"], uploaded_df)
            st.dataframe(predictions.head(100), use_container_width=True)
            st.download_button(
                "Download predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Train the model first to enable predictions.")

with tab_data:
    st.subheader("Dataset Preview")

    if preview_df is not None:
        st.dataframe(preview_df.head(100), use_container_width=True)

        st.subheader("Average Transaction Amount by Category")
        category_amounts = (
            preview_df.groupby("category", as_index=False)["amt"]
            .mean()
            .sort_values("amt", ascending=False)
        )
        st.bar_chart(category_amounts.set_index("category"))
    else:
        st.info("No dataset found in `data/fraudTrain.csv`.")
