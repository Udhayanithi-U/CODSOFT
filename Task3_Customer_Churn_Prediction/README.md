# Customer Churn Prediction

This is my third CodeSoft machine learning task. The goal is to predict whether a customer is likely to leave a subscription or banking service using historical customer details.

I used the customer churn dataset with details such as credit score, geography, gender, age, tenure, account balance, number of products, credit card ownership, active membership status, and estimated salary. The target column is `Exited`, where `1` means the customer churned and `0` means the customer stayed.

## What This Project Does

- Cleans and prepares the churn dataset
- Trains three machine learning models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Compares model performance using accuracy, precision, recall, F1-score, ROC-AUC, and average precision
- Saves the best model
- Provides a Streamlit dashboard for visual presentation
- Supports manual customer churn prediction
- Supports CSV upload for batch prediction
- Includes a Google Colab notebook for running the project online

## Folder Structure

```text
Task3_Customer_Churn_Prediction/
|-- app.py
|-- data/
|   |-- .gitkeep
|-- models/
|   |-- .gitkeep
|-- notebooks/
|   |-- customer_churn_prediction_colab.ipynb
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
Churn_Modelling.csv
```

The dataset is not uploaded to GitHub because it is a data file. Keep it locally inside the `data` folder after extraction.

## Run the Project Locally

Open PowerShell inside this project folder:

```powershell
cd "C:\Users\Udhayanithi\OneDrive\Documents\New project\CODSOFT\Task3_Customer_Churn_Prediction"
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
python src\setup_data.py --zip-path "C:\Users\Udhayanithi\Downloads\archive (3).zip"
```

Train the models:

```powershell
python src\train_model.py
```

The best trained model will be saved here:

```text
models\best_churn_model.joblib
```

## Run the Visual Dashboard

After training the model, start the Streamlit app:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The dashboard has three parts:

- `Overview`: churn rate, customer summary, and model comparison
- `Predict Churn`: manual prediction and CSV upload prediction
- `Data Insights`: charts for churn behavior by customer groups

## Run in Google Colab

Use this notebook:

```text
notebooks/customer_churn_prediction_colab.ipynb
```

In Colab:

1. Upload `archive (3).zip`
2. Extract `Churn_Modelling.csv`
3. Train the models
4. Compare results
5. Save the best model

This is useful when the local system is slow or when the project needs to be shown from a browser.

## Command Line Prediction

After training, predictions can also be generated from a CSV file:

```powershell
python src\predict.py --input data\Churn_Modelling.csv --output reports\churn_predictions.csv
```

The output includes:

- `churn_probability`
- `predicted_churn`

## Model Notes

Customer churn is important because retaining existing customers is usually cheaper than acquiring new ones. In this project, recall and F1-score are useful because the business should not miss too many customers who are actually likely to leave.

The best model is selected using ROC-AUC because it measures how well the model separates churned and retained customers.
