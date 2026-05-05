from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "movie_genre_model.pkl"


def predict_genre(plot_summary: str) -> str:
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([plot_summary])[0]

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba([plot_summary])[0]
        confidence = probabilities.max()
        return f"{prediction} (confidence: {confidence:.2%})"

    return prediction


def main() -> None:
    if not MODEL_PATH.exists():
        print("Model not found. Please run this first:")
        print("python train_model.py")
        return

    print("Movie Genre Predictor")
    print("Paste a movie plot summary below and press Enter.")
    plot_summary = input("Plot summary: ").strip()

    if not plot_summary:
        print("Please enter a plot summary.")
        return

    result = predict_genre(plot_summary)
    print(f"Predicted genre: {result}")


if __name__ == "__main__":
    main()
