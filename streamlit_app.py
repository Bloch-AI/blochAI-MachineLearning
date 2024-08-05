import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# GitHub URL for the dataset
url = 'https://github.com/Bloch-AI/blochAI-MachineLearning/blob/master/wine.xlsx'

# Function to load data from GitHub
@st.cache
def load_data(url):
    st.write(f"Fetching data from URL: {url}")  # Debug statement
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        st.write(f"Response Status Code: {response.status_code}")  # Debug statement
        st.write(f"Response Headers: {response.headers}")  # Debug statement
        file = BytesIO(response.content)
        st.write(f"File size: {len(file.getvalue())} bytes")  # Debug statement
        data = pd.read_excel(file)
        return data
    except Exception as e:
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
data['color'] = data['color'].map({'red': 0, 'white': 1})
data['quality'] = data['quality'].map({
    'bad': 0, 'slightly dissatisfied': 1, 'neutral': 2,
    'good': 3, 'excellent': 4
})

# Feature selection
X = data.drop(['quality'], axis=1)
y = data['quality']

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

