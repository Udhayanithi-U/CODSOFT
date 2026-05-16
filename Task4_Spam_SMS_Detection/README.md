# Spam SMS Detection

This is my fourth CODSOFT machine learning task. The project classifies SMS messages as either spam or legitimate messages.

The dataset contains SMS text messages with labels:

- `ham`: normal message
- `spam`: unwanted or promotional message

I used TF-IDF text features with different classifiers to compare which model works better for spam detection.

## What This Project Includes

- Dataset extraction script
- Text cleaning and preprocessing
- TF-IDF vectorization
- Model training with:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear Support Vector Machine
- Model comparison using accuracy, precision, recall, F1-score, ROC-AUC, and average precision
- Saved best model
- Streamlit visual dashboard
- Manual SMS spam checker
- CSV upload for batch SMS prediction
- Google Colab notebook

## Folder Structure

```text
Task4_Spam_SMS_Detection/
|-- app.py
|-- data/
|   |-- .gitkeep
|-- models/
|   |-- .gitkeep
|-- notebooks/
|   |-- spam_sms_detection_colab.ipynb
|-- reports/
|   |-- .gitkeep
|-- src/
|   |-- predict.py
|   |-- setup_data.py
|   |-- train_model.py
|-- README.md
|-- requirements.txt
```

## Dataset

The dataset zip file should contain:

```text
spam.csv
```

The dataset is not uploaded to GitHub because it is a data file. Use the setup script to extract it locally.

## Run Locally

Open PowerShell inside this project folder:

```powershell
cd "C:\Users\Udhayanithi\OneDrive\Documents\New project\CODSOFT\Task4_Spam_SMS_Detection"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install requirements:

```powershell
pip install -r requirements.txt
```

Extract the dataset:

```powershell
python src\setup_data.py --zip-path "C:\Users\Udhayanithi\Downloads\archive (4).zip"
```

Train the models:

```powershell
python src\train_model.py
```

The best model will be saved here:

```text
models\best_spam_model.joblib
```

## Run the Visual Dashboard

After training, run:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The dashboard includes:

- dataset overview
- spam and ham message distribution
- model comparison
- single SMS spam prediction
- CSV upload for batch prediction
- common words in spam and ham messages

## Run in Google Colab

Use this notebook:

```text
notebooks/spam_sms_detection_colab.ipynb
```

In Colab:

1. Upload `archive (4).zip`
2. Extract `spam.csv`
3. Train the models
4. Compare model results
5. Save the best model

## Command Line Prediction

After training, predict from a CSV file:

```powershell
python src\predict.py --input data\spam.csv --output reports\spam_predictions.csv
```

The output includes:

- `spam_probability`
- `predicted_label`

## Project Note

Spam detection is a text classification problem. In this project, TF-IDF converts SMS text into numerical features, and machine learning models use those features to identify spam patterns.
