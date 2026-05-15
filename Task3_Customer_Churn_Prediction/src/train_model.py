"""Train customer churn prediction models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_PATH = DATA_DIR / "Churn_Modelling.csv"
TARGET = "Exited"
RANDOM_STATE = 42

DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python src/setup_data.py --zip-path <archive.zip>"
        )
    return pd.read_csv(DATASET_PATH)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean_df = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])
    return clean_df.drop(columns=[TARGET]), clean_df[TARGET]


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
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=180,
                        max_depth=12,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=160,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }


def save_reports(results: dict[str, dict]) -> pd.DataFrame:
    REPORTS_DIR.mkdir(exist_ok=True)

    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            }
        )
        report_path = REPORTS_DIR / f"{model_name}_report.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    return comparison


def train(test_size: float = 0.2) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    x, y = prepare_data(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Total rows: {len(df):,}")
    print(f"Training rows: {len(x_train):,}")
    print(f"Testing rows: {len(x_test):,}")
    print(f"Churn rate: {y.mean():.2%}")

    results: dict[str, dict] = {}
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in build_models().items():
        print(f"\nTraining {model_name}...")
        model.fit(x_train, y_train)
        results[model_name] = evaluate_model(model, x_test, y_test)
        fitted_models[model_name] = model

    comparison = save_reports(results)
    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["model"]
    best_model_path = MODELS_DIR / "best_churn_model.joblib"
    joblib.dump(
        {
            "model_name": best_model_name,
            "model": fitted_models[best_model_name],
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        best_model_path,
    )

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {best_model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train customer churn models.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(test_size=args.test_size)
