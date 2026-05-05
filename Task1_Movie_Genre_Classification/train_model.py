from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Genre Classification Dataset"
TRAIN_FILE = DATA_DIR / "train_data.txt"
TEST_FILE = DATA_DIR / "test_data_solution.txt"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_train_data(path: Path) -> pd.DataFrame:
    """Load data formatted as ID ::: TITLE ::: GENRE ::: DESCRIPTION."""
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = [part.strip() for part in line.split(" ::: ")]
            if len(parts) == 4:
                rows.append(
                    {
                        "id": parts[0],
                        "title": parts[1],
                        "genre": parts[2],
                        "description": parts[3],
                    }
                )
    return pd.DataFrame(rows)


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        linewidths=0.2,
        linecolor="white",
        cbar=True,
    )
    plt.title("Movie Genre Classification - Confusion Matrix")
    plt.xlabel("Predicted Genre")
    plt.ylabel("Actual Genre")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    train_df = load_train_data(TRAIN_FILE)
    test_df = load_train_data(TEST_FILE)

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Genres: {train_df['genre'].nunique()}")

    x_train = train_df["description"]
    y_train = train_df["genre"]
    x_test = test_df["description"]
    y_test = test_df["genre"]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    max_features=100000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    )
                ),
            ),
        ]
    )

    print("Training model. This can take a few minutes...")
    model.fit(x_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    joblib.dump(model, MODEL_DIR / "movie_genre_model.pkl")

    with (OUTPUT_DIR / "metrics.txt").open("w", encoding="utf-8") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)

    pd.DataFrame(report_dict).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")

    labels = sorted(train_df["genre"].unique())
    plot_confusion_matrix(y_test, predictions, labels, OUTPUT_DIR / "confusion_matrix.png")

    sample_results = test_df[["id", "title", "genre", "description"]].copy()
    sample_results["predicted_genre"] = predictions
    sample_results.head(100).to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)

    print("Done.")
    print(f"Saved model: {MODEL_DIR / 'movie_genre_model.pkl'}")
    print(f"Saved metrics: {OUTPUT_DIR / 'metrics.txt'}")
    print(f"Saved chart: {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
