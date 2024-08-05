import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import BytesIO
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Custom CSS for styling
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

# Add header
st.markdown('<div class="header"><h1>🍷 Wine Quality Prediction App</h1></div>', unsafe_allow_html=True)

# GitHub URL for the dataset
url = 'https://raw.githubusercontent.com/Bloch-AI/blochAI-MachineLearning/master/wine.xlsx'


# GitHub URL for the dataset
url = 'https://raw.githubusercontent.com/Bloch-AI/blochAI-MachineLearning/master/wine.xlsx'

# Function to load data from GitHub
@st.cache_data
def load_data(url):
    """
    Load data from a given URL.
    Returns a pandas DataFrame.
    """
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
st.write('## Wine Dataset')
st.dataframe(data.head(), height=150)

# Preprocessing
data['color'] = data['color'].map({'red': 0, 'white': 1})

# Enhanced Quality Mapping
quality_mapping = {
    'extremely dissatisfied': 0,
    'moderately dissatisfied': 1,
    'slightly dissatisfied': 2,
    'neutral': 3,
    'slightly satisfied': 4,
    'moderately satisfied': 5,
    'extremely satisfied': 6
}

# Ensure all quality values are mapped correctly
data['quality'] = data['quality'].str.strip().map(quality_mapping)
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
    features = data.drop(['quality', 'color'], axis=1).columns
else:
    target = 'color'
    features = data.drop(['quality', 'color'], axis=1).columns

X = data[features].values
y = data[target].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(f_classif, k=10)  # Select top 10 features
X_selected = selector.fit_transform(X_scaled, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = features[selected_feature_indices]

# Ensure that the test set contains all classes
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
precisions = []
recalls = []
f1_scores = []

for train_index, test_index in skf.split(X_selected, y):
    X_train, X_test = X_selected[train_index], X_selected[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply SMOTE for balancing (only for training data)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Display average metrics
st.write(f'## Model Performance ({prediction_choice})')
st.markdown(f'<div class="result-box">'
            f'### Average Metrics (5-fold cross-validation):<br>'
            f'Accuracy: {np.mean(accuracies):.2f} (+/- {np.std(accuracies):.2f})<br>'
            f'Precision: {np.mean(precisions):.2f} (+/- {np.std(precisions):.2f})<br>'
            f'Recall: {np.mean(recalls):.2f} (+/- {np.std(recalls):.2f})<br>'
            f'F1-score: {np.mean(f1_scores):.2f} (+/- {np.std(f1_scores):.2f})'
            f'</div>', unsafe_allow_html=True)

# Retrain the model on the entire dataset for final predictions
X_resampled, y_resampled = smote.fit_resample(X_selected, y)
final_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X_resampled, y_resampled)

# Make prediction from sidebar input
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
input_selected = selector.transform(input_scaled)
prediction = final_model.predict(input_selected)[0]

if prediction_choice == 'Quality':
    quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}
    predicted_result = quality_mapping_reverse[prediction]
else:
    predicted_result = 'white' if prediction == 1 else 'red'

# Display the prediction result
st.markdown(f'<div class="result-box">### Predicted {prediction_choice}: {predicted_result}</div>', unsafe_allow_html=True)

# Feature importance
importance = final_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
top_features = feature_importance.head(5)

# Plot top 5 feature importances
st.write('### Top 5 Feature Importances')
plt.figure(figsize=(10, 5))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 5 Feature Importances')
plt.gca().invert_yaxis()
st.pyplot(plt)

# ROC Curve
st.write('### ROC Curve')
if prediction_choice == 'Quality':
    y_prob = final_model.predict_proba(X_selected)
    classes_present = np.unique(y)
    quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}

    plt.figure(figsize=(10, 6))
    for i in classes_present:
        fpr, tpr, _ = roc_curve(y == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {quality_mapping_reverse[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)
else:
    y_prob = final_model.predict_proba(X_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Add footer
st.markdown('<div class="footer"><p>© 2024 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p></div>', unsafe_allow_html=True)
