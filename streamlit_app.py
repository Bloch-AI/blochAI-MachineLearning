import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from ucimlrepo import fetch_ucirepo

# Title and description
st.title('Bloch.ai - Machine Learning Demo')
st.title("Wine Color Prediction")
st.write("Predict whether a wine is red or white based on its chemical properties.")

# Fetch dataset
wine_data = fetch_ucirepo(id=186) 

# Data Preparation
X = wine_data.data.features 
y = wine_data.data.targets['type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training (SVM)
model = SVC(kernel='linear') 
model.fit(X_train, y_train)

# User Input
st.header("Enter Wine Properties")
col1, col2 = st.columns(2)
with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
    citric_acid = st.number_input("Citric Acid", min_value=0.0)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
    chlorides = st.number_input("Chlorides", min_value=0.0)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
    density = st.number_input("Density", min_value=0.0)
    pH = st.number_input("pH", min_value=0.0)
    sulphates = st.number_input("Sulphates", min_value=0.0)
    alcohol = st.number_input("Alcohol", min_value=0.0)

# Prediction
if st.button("Predict"):
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                   free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
    input_data_scaled = scaler.transform(input_data)  # Scale input data
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 'red':
        st.success("The wine is predicted to be Red!")
    else:
        st.success("The wine is predicted to be White!")

# Display additional information (optional)
st.header("Model Performance")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
st.text(report)
