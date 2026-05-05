from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "movie_genre_model.pkl"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.txt"
CONFUSION_MATRIX_PATH = BASE_DIR / "outputs" / "confusion_matrix.png"


model = joblib.load(MODEL_PATH)


def predict_genre(plot_summary):
    if not plot_summary or not plot_summary.strip():
        return "Please enter a movie plot summary.", pd.DataFrame()

    prediction = model.predict([plot_summary])[0]
    probabilities = model.predict_proba([plot_summary])[0]
    classes = model.named_steps["classifier"].classes_

    confidence_df = pd.DataFrame(
        {
            "Genre": classes,
            "Confidence (%)": probabilities * 100,
        }
    )
    confidence_df = confidence_df.sort_values("Confidence (%)", ascending=False).head(5)
    confidence_df["Confidence (%)"] = confidence_df["Confidence (%)"].round(2)

    return f"Predicted Genre: {prediction.title()}", confidence_df


metrics_text = "Run train_model.py to generate the metrics report."
if METRICS_PATH.exists():
    metrics_text = METRICS_PATH.read_text(encoding="utf-8")


with gr.Blocks(title="Movie Genre Classification") as app:
    gr.Markdown("# Movie Genre Classification App")
    gr.Markdown(
        "Enter a movie plot summary and this machine learning model will predict its genre."
    )

    with gr.Row():
        with gr.Column(scale=1):
            plot_input = gr.Textbox(
                label="Movie Plot Summary",
                lines=8,
                placeholder=(
                    "Example: A detective investigates a mysterious murder in a small "
                    "town and discovers a hidden conspiracy."
                ),
            )
            predict_button = gr.Button("Predict Genre", variant="primary")

        with gr.Column(scale=1):
            prediction_output = gr.Textbox(label="Prediction")
            confidence_output = gr.Dataframe(label="Top 5 Genre Confidence Scores")

    predict_button.click(
        fn=predict_genre,
        inputs=plot_input,
        outputs=[prediction_output, confidence_output],
    )

    gr.Markdown("## Confusion Matrix")
    if CONFUSION_MATRIX_PATH.exists():
        gr.Image(value=str(CONFUSION_MATRIX_PATH), label="Model Confusion Matrix")
    else:
        gr.Markdown("Confusion matrix not found. Run `train_model.py` first.")

    gr.Markdown("## Model Evaluation Report")
    gr.Textbox(value=metrics_text, label="Metrics", lines=18)


if __name__ == "__main__":
    app.launch()
