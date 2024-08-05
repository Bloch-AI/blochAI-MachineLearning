import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# GitHub URL for the dataset
url = 'https://raw.githubusercontent.com/Bloch-AI/blochAI-MachineLearning/master/wine.xlsx'

# Function to load data from GitHub
@st.cache_data
def load_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        file = BytesIO(response.content)
        data = pd.read_excel(file, engine='openpyxl')
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
data = load_data(url)
if data is None:
    st.stop()

# Display dataset
st.title('Wine Quality Prediction App')
st.write('## Wine Dataset')
st.write(data.head())

st.write(data.info())

# Preprocessing
st.write('## Preprocessing')
data['color'] = data['color'].map({'red': 0, 'white': 1})

# Enhanced Quality Mapping
quality_mapping = {
    'extremly dissatisfied': 0,
    'moderately dissatisfied': 1,
    'slightly dissatisfied': 2,
    'neutral': 3,
    'slightly satisfied': 4,
    'moderately satisfied': 5,
    'extremly satisfied': 6
}

data['quality'] = data['quality'].astype(str).map(quality_mapping)
data.dropna(subset=['quality'], inplace=True)

# Sidebar for parameter selection and prediction input
with st.sidebar:
    st.write('## Model Parameters')
    test_size = st.slider('Test Size', 0.1, 0.5, 0.2)

    # Input features for prediction
    st.write('## Predict Wine Quality')
    user_input = {}
    for feature in data.drop(['quality'], axis=1).columns:
        user_input[feature] = st.number_input(f'{feature}', float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

# Feature selection (X converted to NumPy array)
X = data.drop(['quality'], axis=1).values
y = data['quality'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the model
st.write('## Training Model')
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f'### Accuracy: {accuracy:.2f}')

# Feature importance
st.write('## Feature Importance')
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': data.drop(['quality'], axis=1).columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
st.write(feature_importance)

# Make prediction from sidebar input
input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}  
predicted_quality = quality_mapping_reverse[prediction]
st.sidebar.write(f'### Predicted Quality: {predicted_quality}')  # Display prediction in sidebar
