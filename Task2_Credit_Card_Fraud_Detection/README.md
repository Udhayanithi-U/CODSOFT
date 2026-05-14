# Credit Card Fraud Detection

This is a machine learning project for detecting fraudulent credit card transactions. It compares Logistic Regression, Decision Tree, and Random Forest models, then provides a Streamlit web app so the project can be shown visually during presentations, internships, or hackathons.

## Main Features

- Trains fraud detection models on transaction data
- Compares Logistic Regression, Decision Tree, and Random Forest
- Handles imbalanced fraud data with precision, recall, F1-score, ROC-AUC, and average precision
- Saves the best trained model
- Provides a visual Streamlit dashboard
- Supports single transaction fraud checking
- Supports CSV upload for batch fraud prediction
- Includes a Google Colab notebook for cloud execution

## Project Structure

```text
credit-card-fraud-detection/
|-- app.py
|-- data/
|   |-- .gitkeep
|-- models/
|   |-- .gitkeep
|-- notebooks/
|   |-- credit_card_fraud_detection_colab.ipynb
|-- reports/
|   |-- .gitkeep
|-- src/
|   |-- predict.py
|   |-- setup_data.py
|   |-- train_model.py
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## Dataset

Use the credit card fraud dataset zip file that contains:

- `fraudTrain.csv`
- `fraudTest.csv`

The dataset files are large, so they are not pushed to GitHub. After cloning or copying the project, extract the dataset locally.

## Run Locally

### 1. Open the Project Folder

```powershell
cd "C:\Users\Udhayanithi\Documents\Codex\2026-05-14\credit-card-fraud-detection-build-a"
```

### 2. Activate Virtual Environment

```powershell
.\.venv\Scripts\activate
```

If the virtual environment does not exist, create it:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Extract Dataset

```powershell
python src\setup_data.py --zip-path "C:\Users\Udhayanithi\Downloads\archive (2).zip"
```

### 4. Train the Model

Quick test run:

```powershell
python src\train_model.py --sample-frac 0.02
```

Full training:

```powershell
python src\train_model.py
```

The best model is saved to:

```text
models\best_fraud_model.joblib
```

### 5. Run the Visual Website

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Website Pages

The Streamlit app has three sections:

- `Project Dashboard`: shows model performance and fraud data charts
- `Fraud Prediction`: predicts fraud risk for one transaction or uploaded CSV files
- `Dataset View`: previews the dataset and category-based transaction patterns

## Run in Google Colab

Open this notebook:

```text
notebooks/credit_card_fraud_detection_colab.ipynb
```

In Colab:

1. Upload `archive (2).zip`
2. Install dependencies
3. Extract `fraudTrain.csv` and `fraudTest.csv`
4. Train the models
5. Save the best model

The notebook is useful if your laptop is slow or you want to run the ML training online.

## Command Line Prediction

After training, generate predictions from the test data:

```powershell
python src\predict.py --input data\fraudTest.csv --output reports\test_predictions.csv
```

The output file includes:

- `fraud_probability`
- `predicted_is_fraud`

## Model Results

Reports are saved in:

```text
reports\
```

Important generated files:

- `reports\model_comparison.csv`
- `reports\random_forest_report.json`
- `reports\decision_tree_report.json`
- `reports\logistic_regression_report.json`

## GitHub Notes

Do not push large generated files to GitHub:

- `data\fraudTrain.csv`
- `data\fraudTest.csv`
- `models\best_fraud_model.joblib`
- `reports\test_predictions.csv`

These files are already ignored in `.gitignore`.

## Internship Task Summary

This project satisfies the CodeSoft task by building a fraud detection model using credit card transaction data and experimenting with Logistic Regression, Decision Tree, and Random Forest algorithms to classify transactions as fraudulent or legitimate.
