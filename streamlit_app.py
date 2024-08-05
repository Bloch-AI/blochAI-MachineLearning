import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

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
    prediction_choice = st.radio("Choose what to predict", ('Quality', 'Color'))
    test_size = st.slider('Test Size', 0.1, 0.5, 0.2)

    # Input features for prediction
    st.write(f'## Predict Wine {prediction_choice}')
    user_input = {}
    for feature in data.drop(['quality', 'color'], axis=1).columns:
        user_input[feature] = st.number_input(f'{feature}', float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

# Feature selection
if prediction_choice == 'Quality':
    target = 'quality'
    features = data.drop(['quality'], axis=1).columns
else:
    target = 'color'
    features = data.drop(['color'], axis=1).columns

X = data[features].values
y = data[target].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the model
st.write(f'## Training Model to Predict {prediction_choice}')
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f'### Accuracy: {accuracy:.2f}')

# Feature importance
st.write('## Feature Importance')
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
st.write(feature_importance)

# Make prediction from sidebar input
if prediction_choice == 'Quality':
    input_df = pd.DataFrame([user_input])
else:
    user_input['color'] = st.number_input('color', 0, 1, 1)  # Adding the color feature when predicting quality
    input_df = pd.DataFrame([user_input])

# Ensure input_df has all the necessary features
input_df = input_df.reindex(columns=features, fill_value=0)

prediction = model.predict(input_df)[0]

if prediction_choice == 'Quality':
    quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}
    predicted_result = quality_mapping_reverse[prediction]
else:
    predicted_result = 'white' if prediction == 1 else 'red'

st.sidebar.write(f'### Predicted {prediction_choice}: {predicted_result}')

# ROC Curve for Quality prediction
if prediction_choice == 'Quality':
    st.write('## ROC Curve')
    y_prob = model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(quality_mapping)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'purple']
    for i, color in zip(range(len(quality_mapping)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {quality_mapping_reverse[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    st.pyplot(plt)
