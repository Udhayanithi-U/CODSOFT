"""Generate fraud predictions with the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train_model import MODELS_DIR, add_features


DEFAULT_MODEL_PATH = MODELS_DIR / "best_fraud_model.joblib"


def predict(input_path: Path, output_path: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with python src/train_model.py"
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]

    raw_data = pd.read_csv(input_path)
    features = add_features(raw_data)
    if "is_fraud" in features.columns:
        features = features.drop(columns=["is_fraud"])

    fraud_probability = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    output = raw_data.copy()
    output["fraud_probability"] = fraud_probability
    output["predicted_is_fraud"] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraudulent transactions.")
    parser.add_argument("--input", type=Path, required=True, help="Input transaction CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/predictions.csv"),
        help="Where to save predictions.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model joblib file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.input, args.output, args.model)
