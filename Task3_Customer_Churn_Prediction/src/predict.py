"""Generate churn predictions using the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train_model import DROP_COLUMNS, MODELS_DIR, TARGET


DEFAULT_MODEL_PATH = MODELS_DIR / "best_churn_model.joblib"


def prepare_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])
    if TARGET in features.columns:
        features = features.drop(columns=[TARGET])
    return features


def predict(input_path: Path, output_path: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with python src/train_model.py"
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]

    raw_df = pd.read_csv(input_path)
    features = prepare_prediction_data(raw_df)

    output = raw_df.copy()
    output["churn_probability"] = model.predict_proba(features)[:, 1]
    output["predicted_churn"] = model.predict(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn from a CSV file.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/churn_predictions.csv"),
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
