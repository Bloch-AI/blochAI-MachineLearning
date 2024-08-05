# Import necessary libraries
import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.pipeline import Pipeline

# Custom CSS for styling the Streamlit app
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .block-container {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f5;
        padding: 2rem;
    }
    .prediction-box, .result-box {
        padding: 10px;
        border: 2px solid black;
        background-color: lightyellow;
        text-align: center;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #333;
    }
    .feature-importance {
        margin-top: 20px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #000000;
        color: white;
        text-align: center;
        padding: 1rem;
    }
    .header {
        width: 100%;
        background-color: #ffffff;
        color: white;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Add header to the app
st.markdown('<div class="header"><h1>🍷 Wine Quality Prediction App</h1></div>', unsafe_allow_html=True)

# GitHub URL for the wine dataset
url = 'https://raw.githubusercontent.com/Bloch-AI/blochAI-MachineLearning/master/wine.xlsx'

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

# Load the dataset
data = load_data(url)
if data is None:
    st.stop()

# Display the first few rows of the dataset
st.write('## Wine Dataset')
st.dataframe(data.head(), height=150)

# Preprocess the data
data['color'] = data['color'].map({'red': 0, 'white': 1})

quality_mapping = {
    'extremely dissatisfied': 0,
    'moderately dissatisfied': 1,
    'slightly dissatisfied': 2,
    'neutral': 3,
    'slightly satisfied': 4,
    'moderately satisfied': 5,
    'extremely satisfied': 6
}

data['quality'] = data['quality'].str.strip().map(quality_mapping)
data.dropna(subset=['quality'], inplace=True)

# Sidebar for user inputs
with st.sidebar:
    st.write('## Model Parameters')
    prediction_choice = st.radio("Choose what to predict", ('Quality', 'Color'))
    model_choice = st.radio("Choose model", ('Random Forest', 'XGBoost'))
    test_size = st.slider('Test Size', 0.1, 0.5, 0.2)

    st.write(f'## Predict Wine {prediction_choice}')
    user_input = {}
    for feature in data.drop(['quality', 'color'], axis=1).columns:
        user_input[feature] = st.number_input(f'{feature}', float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))

# Prepare features and target based on user's choice
if prediction_choice == 'Quality':
    target = 'quality'
    features = data.drop(['quality', 'color'], axis=1).columns
else:
    target = 'color'
    features = data.drop(['quality', 'color'], axis=1).columns

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Create pipelines for both models
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42))
])

xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
])

# Choose the appropriate pipeline
if model_choice == 'Random Forest':
    model_pipeline = rf_pipeline
else:
    model_pipeline = xgb_pipeline

# Train the model with a progress bar
with st.spinner(f'Training {model_choice} model...'):
    model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Display model performance metrics
st.write(f'## Model Performance ({model_choice} - {prediction_choice})')
st.markdown(f'<div class="result-box">'
            f'### Metrics:<br>'
            f'Accuracy: {accuracy:.2f}<br>'
            f'Precision: {precision:.2f}<br>'
            f'Recall: {recall:.2f}<br>'
            f'F1-score: {f1:.2f}'
            f'</div>', unsafe_allow_html=True)

# Make prediction based on user input
input_df = pd.DataFrame([user_input])
prediction = model_pipeline.predict(input_df)[0]

# Interpret the prediction
if prediction_choice == 'Quality':
    quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}
    predicted_result = quality_mapping_reverse[prediction]
else:
    predicted_result = 'white' if prediction == 1 else 'red'

# Display the prediction result
st.markdown(f'<div class="result-box">### Predicted {prediction_choice}: {predicted_result}</div>', unsafe_allow_html=True)

# Calculate and display feature importances
if model_choice == 'Random Forest':
    importance = model_pipeline.named_steps['rf'].feature_importances_
else:  # XGBoost
    importance = model_pipeline.named_steps['xgb'].feature_importances_

feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
top_features = feature_importance.head(5)

st.write('### Top 5 Feature Importances')
plt.figure(figsize=(10, 5))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'Top 5 Feature Importances ({model_choice})')
plt.gca().invert_yaxis()
st.pyplot(plt)

# Plot ROC curve
st.write('### ROC Curve')
if prediction_choice == 'Quality':
    # Multi-class ROC curve for Quality prediction
    y_prob = model_pipeline.predict_proba(X_test)
    classes_present = np.unique(y)
    quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}

    plt.figure(figsize=(10, 6))
    for i in classes_present:
        fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {quality_mapping_reverse[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC Curve ({model_choice})')
    plt.legend(loc="lower right")
    st.pyplot(plt)
else:
    # Binary ROC curve for Color prediction
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_choice})')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Add footer
st.markdown('<div class="footer"><p>© 2024 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p></div>', unsafe_allow_html=True)
