# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease using patient medical data. This project was completed as part of the CodeAlpha Machine Learning Internship.

## Overview

The project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

Three machine learning classification models were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

The project includes data preprocessing, model training, evaluation, visualization, model saving, and an interactive Streamlit web application.

## Dataset

The dataset used is the Cleveland subset of the UCI Heart Disease dataset.

The dataset contains patient information such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- ST segment slope
- Number of major vessels
- Thalassemia

The target variable represents whether heart disease is present.

## Machine Learning Models

### Logistic Regression

Logistic Regression was used as a baseline classification model for predicting the binary target.

### Support Vector Machine

SVM was used to find a decision boundary between the two target classes.

### Random Forest

Random Forest uses multiple decision trees to make predictions and can capture nonlinear relationships in the data.

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

The project also includes Random Forest feature importance analysis.

## Streamlit Application

An interactive Streamlit application was created for testing the trained models.

The application allows users to:

- Enter patient information
- Select a machine learning model
- Generate a prediction
- View prediction probabilities
- Compare the different model approaches

> **Important:** This application is intended for educational and demonstration purposes only. It is not a medical diagnostic tool.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Streamlit

## Project Structure

```text
CodeAlpha_Disease_Prediction/
│
├── data/
│   └── processed.cleveland.data
│
├── disease_prediction.ipynb
├── app.py
│
├── logistic_regression_model.pkl
├── svm_model.pkl
├── random_forest_model.pkl
├── scaler.pkl
│
├── model_results.csv
├── feature_importance.csv
├── requirements.txt
├── README.md
└── .gitignore
```
# How to Run
1. Install the required packages
```
pip install -r requirements.txt
```
2. Open the Jupyter Notebook
```
jupyter notebook
```
Open:
```
disease_prediction.ipynb
```
and run the cells in order.

3. Run the Streamlit Application

From the project directory:
```
streamlit run app.py
```
The application will open in a web browser.

## Disclaimer

This project is an educational machine learning demonstration and should not be used for medical diagnosis or clinical decision-making.

