"""Generate spam predictions from SMS messages."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train_model import MODELS_DIR, clean_text


DEFAULT_MODEL_PATH = MODELS_DIR / "best_spam_model.joblib"


def find_message_column(df: pd.DataFrame) -> str:
    for column in ["message", "sms", "text", "v2"]:
        if column in df.columns:
            return column
    raise ValueError("Input CSV must contain one of these columns: message, sms, text, v2")


def predict(input_path: Path, output_path: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with python src/train_model.py"
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]

    df = pd.read_csv(input_path, encoding="latin-1")
    message_column = find_message_column(df)
    clean_messages = df[message_column].map(clean_text)

    output = df.copy()
    output["spam_probability"] = model.predict_proba(clean_messages)[:, 1]
    output["predicted_label"] = [
        "spam" if prediction == 1 else "ham" for prediction in model.predict(clean_messages)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict spam messages from a CSV file.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/spam_predictions.csv"),
        help="Output prediction CSV path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Trained model path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.input, args.output, args.model)
