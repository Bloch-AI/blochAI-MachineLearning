import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

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

# Convert quality to numeric and handle missing values
data['quality'] = pd.to_numeric(data['quality'], errors='coerce')
data.dropna(subset=['quality'], inplace=True)

# Check if quality contains any infinite values (should be handled by previous steps)
if np.isinf(data['quality']).any():
    st.error("Quality column contains infinite values.")
    st.stop()


# Quality Mapping
def map_quality(x):
    if x <= 4:
        return 'bad'
    elif x <= 6:
        return 'slightly dissatisfied'
    elif x == 7:
        return 'neutral'
    elif x <= 8:
        return 'good'
    else:
        return 'excellent'


data['quality'] = data['quality'].apply(map_quality)
data['quality'] = data['quality'].map(
    {'bad': 0, 'slightly dissatisfied': 1, 'neutral': 2, 'good': 3, 'excellent': 4}
)

# Feature selection
X = data.drop(['quality'], axis=1)
y = data['quality'].values # converting to np array

# Split the data
st.write('## Train/Test Split')
test_size = st.slider('Select test size', 0.1, 0.5, 0.2)
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
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
st.write(feature_importance)

# Predict on user input
st.write('## Predict Wine Quality')
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(f'{feature}', float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
st.write(f'### Predicted Quality: {prediction}')

