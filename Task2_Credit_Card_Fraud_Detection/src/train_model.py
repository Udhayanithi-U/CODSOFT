"""Train and compare credit card fraud detection models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "is_fraud"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
    "age",
    "hour",
    "day_of_week",
    "month",
]

CATEGORICAL_FEATURES = [
    "category",
    "gender",
    "state",
]

DROP_COLUMNS = [
    "Unnamed: 0",
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "job",
    "merchant",
    "trans_num",
    "trans_date_trans_time",
    "dob",
]


def load_data(sample_frac: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "fraudTrain.csv"
    test_path = DATA_DIR / "fraudTest.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing dataset files. Run: python src/setup_data.py --zip-path <archive.zip>"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=RANDOM_STATE)
        test_df = test_df.sample(frac=sample_frac, random_state=RANDOM_STATE)

    return train_df, test_df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    transaction_time = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    birth_date = pd.to_datetime(df["dob"], errors="coerce")

    df["hour"] = transaction_time.dt.hour
    df["day_of_week"] = transaction_time.dt.dayofweek
    df["month"] = transaction_time.dt.month
    df["age"] = ((transaction_time - birth_date).dt.days / 365.25).round(1)

    return df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prepared = add_features(df)
    return prepared.drop(columns=[TARGET]), prepared[TARGET]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        fraud_scores = model.predict_proba(x_test)[:, 1]
    else:
        fraud_scores = predictions

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_test, fraud_scores)),
        "average_precision": float(average_precision_score(y_test, fraud_scores)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
    }


def save_reports(results: dict[str, dict]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    summary_rows = []
    for model_name, metrics in results.items():
        summary_rows.append(
            {
                "model": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            }
        )

        report_path = REPORTS_DIR / f"{model_name}_report.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = pd.DataFrame(summary_rows).sort_values(
        "average_precision", ascending=False
    )
    summary.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(summary.to_string(index=False))


def train(sample_frac: float | None = None) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    train_df, test_df = load_data(sample_frac=sample_frac)
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    print(f"Training rows: {len(x_train):,}")
    print(f"Testing rows: {len(x_test):,}")
    print(f"Fraud rate in training data: {np.mean(y_train):.4%}")

    models = build_models()
    results: dict[str, dict] = {}
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(x_train, y_train)
        results[model_name] = evaluate_model(model, x_test, y_test)
        fitted_models[model_name] = model

    save_reports(results)

    best_model_name = max(
        results,
        key=lambda name: results[name]["average_precision"],
    )
    best_model_path = MODELS_DIR / "best_fraud_model.joblib"
    joblib.dump(
        {
            "model_name": best_model_name,
            "model": fitted_models[best_model_name],
            "features": {
                "numeric": NUMERIC_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
            },
        },
        best_model_path,
    )

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {best_model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction of train/test data to use for a quicker run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(sample_frac=args.sample_frac)
