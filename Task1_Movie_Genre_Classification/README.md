@'
# Movie Genre Classification

This project was completed as Task 1 of my CodeSoft Machine Learning Internship.

## Objective

Build a machine learning model that predicts the genre of a movie based on its plot summary or other textual information.

## Dataset

The dataset contains movie titles, genres, and plot descriptions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Gradio

## Methodology

1. Loaded the movie genre dataset.
2. Preprocessed the text-based plot summaries.
3. Converted text data into numerical features using TF-IDF.
4. Trained a Logistic Regression classification model.
5. Evaluated the model using accuracy, precision, recall, and F1-score.
6. Built a simple Gradio web app for genre prediction.

## Results

The model achieved an accuracy of approximately 59-61% on the test dataset.

## How To Run

```bash
pip install -r requirements.txt
python train_model.py
python app.py
