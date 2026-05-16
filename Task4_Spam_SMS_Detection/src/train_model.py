"""Train SMS spam detection models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_PATH = DATA_DIR / "spam.csv"
RANDOM_STATE = 42


def clean_text(message: str) -> str:
    message = str(message).lower()
    message = re.sub(r"http\S+|www\S+", " url ", message)
    message = re.sub(r"\d+", " number ", message)
    message = re.sub(r"[^a-z\s]", " ", message)
    return re.sub(r"\s+", " ", message).strip()


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python src/setup_data.py --zip-path <archive.zip>"
        )

    df = pd.read_csv(DATASET_PATH, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].str.lower().str.strip()
    df["spam"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].map(clean_text)
    return df.dropna(subset=["spam", "clean_message"])


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline(
            steps=[
                ("tfidf", build_vectorizer()),
                ("model", MultinomialNB(alpha=0.2)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("tfidf", build_vectorizer()),
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
        "linear_svm": Pipeline(
            steps=[
                ("tfidf", build_vectorizer()),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict:
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

    comparison = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    return comparison


def train(test_size: float = 0.2) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["spam"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["spam"],
    )

    print(f"Total messages: {len(df):,}")
    print(f"Training messages: {len(x_train):,}")
    print(f"Testing messages: {len(x_test):,}")
    print(f"Spam rate: {df['spam'].mean():.2%}")

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
    best_model_path = MODELS_DIR / "best_spam_model.joblib"
    joblib.dump(
        {
            "model_name": best_model_name,
            "model": fitted_models[best_model_name],
        },
        best_model_path,
    )

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {best_model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMS spam detection models.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of messages used for testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(test_size=args.test_size)
