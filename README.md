# 💻 Bloch.ai - Machine Learning Demo

## 🍷 Wine Quality Prediction App
This Streamlit app demonstrates how machine learning works, focusing on simple classification algorithms for predicting wine quality and colour. Users can experiment with data, adjust parameters, and select different models to observe how these changes affect the model's performance.

## Features
Predict wine quality or colour using machine learning models
Choose from different models: Random Forest, XGBoost, or Decision Tree
Adjust model parameters and hyperparameters
Visualise model performance metrics and feature importance
Interactive user input for wine characteristics
Display of ROC curves for model evaluation

## How to Use
Select what to predict: Quality or Colour
Choose a machine learning model
Adjust test size and model hyperparameters
Input custom wine characteristics or use default values
View model performance metrics, feature importance, and ROC curves

## Dataset
The app uses the UCI Machine Learning Repository wine dataset, which includes various chemical components of wines along with their quality ratings and colours.
Model Performance Metrics
The app displays the following performance metrics:

Accuracy
Precision
Recall
F1-score

## Visualisations
Top 5 Most Important Features bar chart
ROC Curve (binary for Colour prediction, multi-class for Quality prediction)

## Technologies Used
Python
Streamlit
Pandas
Scikit-learn
XGBoost
Matplotlib

## Running the App
To run the app locally:
Clone this repository
Install the required packages: pip install -r requirements.txt
Run the Streamlit app: streamlit run streamlit_app.py

## Further Reading
Streamlit Documentation
Scikit-learn Documentation
XGBoost Documentation
UCI Wine Quality Dataset

## About
This app was created by Bloch AI LTD. For more information, visit www.bloch.ai.
© 2024 Bloch AI LTD - All Rights Reserved.
