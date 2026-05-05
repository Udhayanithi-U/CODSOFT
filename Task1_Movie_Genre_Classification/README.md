---
title: Movie Genre Classification
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# Movie Genre Classification

This project predicts a movie genre from a plot summary using a machine learning pipeline:

- Text feature extraction: TF-IDF
- Classifier: Logistic Regression
- Evaluation: accuracy, classification report, confusion matrix

This is suitable for the CodeSoft machine learning internship Task 1.

## Project Structure

```text
movie-genre-classification/
├── Genre Classification Dataset/
│   ├── description.txt
│   ├── train_data.txt
│   ├── test_data.txt
│   └── test_data_solution.txt
├── models/
├── outputs/
├── train_model.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## What This Project Does

The model reads movie plot summaries and learns patterns that connect words with genres.
For example, words related to police, murder, and investigation may point toward crime or thriller, while words about family and relationships may point toward drama.

## Easiest Platform For Beginners

Use **Google Colab** if you do not know VS Code yet.

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File > New notebook**.
3. Upload this project folder or upload `archive.zip`.
4. Run the installation command:

```bash
!pip install pandas scikit-learn joblib matplotlib seaborn
```

5. If you uploaded `archive.zip`, unzip it:

```bash
!unzip archive.zip
```

6. Upload `train_model.py` and run:

```bash
!python train_model.py
```

7. To test your own plot:

```bash
!python predict.py
```

If you are using this folder on your laptop, follow the local steps below.

## Run On Your Laptop

Open Command Prompt or PowerShell inside this folder and run:

```bash
python -m pip install -r requirements.txt
python train_model.py
python predict.py
```

## Run As A Web App

After training the model, run:

```bash
python app.py
```

The app lets you paste a movie plot summary, predict the genre, and view the model report and confusion matrix.

## Deploy Permanently

The easiest free deployment platform for this project is Hugging Face Spaces.

1. Create a new Space on Hugging Face.
2. Choose **Gradio** as the SDK.
3. Upload this project folder.
4. Hugging Face will install `requirements.txt` and run `app.py`.

After training, these files are created:

- `models/movie_genre_model.pkl`
- `outputs/metrics.txt`
- `outputs/classification_report.csv`
- `outputs/confusion_matrix.png`
- `outputs/sample_predictions.csv`

## Algorithm Used

The project uses a Scikit-learn pipeline:

1. `TfidfVectorizer` converts plot summaries into numerical features.
2. `LogisticRegression` learns how those features map to genres.
3. The trained model predicts the genre for unseen plot summaries.

## Sample Prediction

After running `python predict.py`, paste a plot summary like:

```text
A detective investigates a mysterious murder in a small town and discovers a hidden conspiracy.
```

The program will print a predicted genre.

## GitHub Upload Steps

A repository is just a project folder saved on GitHub.

Simple method:

1. Go to [GitHub](https://github.com/).
2. Create an account or sign in.
3. Click the **+** button in the top-right corner.
4. Click **New repository**.
5. Repository name: `movie-genre-classification`
6. Keep it **Public**.
7. Click **Create repository**.
8. Click **uploading an existing file**.
9. Upload the files from this folder.
10. Click **Commit changes**.

Do not upload `models/movie_genre_model.pkl` if GitHub says the file is too large. The code can recreate it by running `train_model.py`.

## Internship Explanation

This project builds a machine learning model to classify movie genres based on plot summaries. The dataset contains movie titles, genres, and descriptions. Text data is converted into numerical TF-IDF features, and a Logistic Regression classifier is trained on those features. The model is evaluated using the provided test solution file, and the results are saved in the `outputs` folder.
