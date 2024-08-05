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

# Preprocessing
st.write('## Preprocessing')
st.write(f"Initial data shape: {data.shape}")

# Map color column
data['color'] = data['color'].map({'red': 0, 'white': 1})

# Convert quality to numeric and handle missing values
st.write(f"Unique values in quality before conversion: {data['quality'].unique()}")
data['quality'] = pd.to_numeric(data['quality'], errors='coerce')
st.write(f"Unique values in quality after conversion to numeric: {data['quality'].unique()}")
data.dropna(subset=['quality'], inplace=True)
st.write(f"Data shape after handling missing values: {data.shape}")

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

if not data['quality'].empty:
    data['quality'] = data['quality'].apply(map_quality)
    data['quality'] = data['quality'].map(
        {'bad': 0, 'slightly dissatisfied': 1, 'neutral': 2, 'good': 3, 'excellent': 4}
    )
else:
    st.error("No valid quality data available after conversion.")
    st.stop()

st.write(f"Data shape after mapping quality: {data.shape}")

# Check for any remaining NaNs or infinite values
st.write(f"Number of NaNs in each column: {data.isna().sum()}")
st.write(f"Number of infinite values in each column: {np.isinf(data).sum().sum()}")

# Feature selection
X = data.drop(['quality'], axis=1).values
y = data['quality'].values

# Debug statements for checking shapes
st.write(f"Shape of X: {X.shape}")
st.write(f"Shape of y: {y.shape}")

# Split the data
st.write('## Train/Test Split')
test_size = st.slider('Select test size', 0.1, 0.5, 0.2)
st.write(f"Selected test size: {test_size}")

# Additional debug information
st.write(f"Type of X: {type(X)}, Type of y: {type(y)}")
st.write(f"First 5 elements of X: {X[:5]}")
st.write(f"First 5 elements of y: {y[:5]}")

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.write(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    st.write(f"Train target shape: {y_train.shape}, Test target shape: {y_test.shape}")
except ValueError as e:
    st.error(f"Error during train/test split: {e}")
    st.stop()

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

# Predict on user input
st.write('## Predict Wine Quality')
user_input = {}
for feature in data.drop(['quality'], axis=1).columns:
    user_input[feature] = st.number_input(f'{feature}', float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
st.write(f'### Predicted Quality: {prediction}')
