#**********************************************
# Wine Quality Prediction App
# Version 1
# 1st September 2024 (Updated: 2nd February 2025)
# Jamie Crossman-Smith
# jamie@bloch.ai
#**********************************************
# This Python code creates an interactive educational web application using Streamlit.
# The application demonstrates how machine learning can be used for wine quality (and colour)
# prediction using various models. Users can experiment with the data, adjust parameters and
# select different models (Random Forest, XGBoost, Decision Tree) to observe how these changes
# affect the model's performance.
#
# The app displays:
# - The wine dataset (with a brief explanation of its columns)
# - Model performance metrics such as accuracy, precision, recall, F1-score and ROC curves
# - Feature importance charts (for tree‑based models)
#
# Educational explanations are provided throughout the app (via sidebar expanders and info boxes)
# to help users understand the model parameters, hyperparameters and how to interpret the visual outputs.
#
# This app is intended to enhance understanding of machine learning in a practical, interactive manner.
#**********************************************

# Import necessary libraries for building the Streamlit app and handling data
import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import BytesIO

# Import libraries for machine learning and evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import xgboost as xgb

# Custom CSS for styling the app
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
    .header {
        width: 100%;
        background-color: #ffffff;
        color: black;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .explanation-box {
        background-color: #E6F3FF;
        border: 2px solid black;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .intro-box {
        background-color: #FFEBCC;
        border: 2px solid black;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 10px;
        border: 2px solid black;
        background-color: lightyellow;
        text-align: center;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Functions for custom explanation and intro boxes
def explanation_box(text):
    st.markdown(f'<div class="explanation-box">{text}</div>', unsafe_allow_html=True)

def intro_box(text):
    st.markdown(f'<div class="intro-box">{text}</div>', unsafe_allow_html=True)

# Cache the data loading function for efficiency
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

def main():
    # Header & Introduction
    st.markdown('<div class="header"><h1>🍷 Wine Quality Prediction App</h1></div>', unsafe_allow_html=True)
    intro_box("""
    This app demonstrates how machine learning works, focusing on wine quality and colour prediction.
    Experiment with the data, adjust parameters and select different models to observe how these changes
    affect model performance. By interacting with these elements, you'll gain practical insights into the workings
    of machine learning systems.
    """)

    # Data Loading
    url = 'https://raw.githubusercontent.com/Bloch-AI/blochAI-MachineLearning/master/wine.xlsx'
    data = load_data(url)
    if data is None:
        st.stop()

    st.write('## Wine Dataset')
    explanation_box("""
    This dataset, drawn from the UCI Machine Learning Repository wine dataset, contains various chemical
    attributes of wines along with class columns for Quality and Colour. The data has been preprocessed for modelling.
    """)
    st.dataframe(data.head(), height=150)

    # Data Preprocessing
    # Map wine colour to numeric values (UK English: "Colour")
    data['color'] = data['color'].map({'red': 0, 'white': 1})
    # Map quality text to numeric values – strip spaces and map; drop rows where quality is missing
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

    # Sidebar: Model Parameters & Hyperparameters
    st.sidebar.header('Model Parameters')
    with st.sidebar.expander("Learn about Model Parameters"):
        st.markdown("""
        **Prediction Target:**  
        Choose whether to predict wine Quality (a multi-class problem) or Colour (binary classification).  
        This selection determines which column the model will try to predict.

        **Model Choice:**  
        Select from three models:
        - **Random Forest:** An ensemble of decision trees for robust predictions.
        - **XGBoost:** An advanced boosting technique that builds trees sequentially.
        - **Decision Tree:** A simple tree-based model for clear, interpretable decisions.
        """)
    # Use UK English: 'Colour' (not 'Color')
    prediction_choice = st.sidebar.radio("Choose what to predict", ('Quality', 'Colour'))

    st.sidebar.header('Model Hyperparameters')
    with st.sidebar.expander("Learn about Hyperparameters"):
        st.markdown("""
        **Test Set Size:**  
        This determines the fraction of the data used for testing. A larger test size means less data for training.

        **For Random Forest & XGBoost:**  
        - **Number of Trees:** More trees can improve performance but increase processing time.
        - **Max Depth:** Controls how deep each tree can grow; deeper trees may capture more complex patterns but risk overfitting.
        
        **For Decision Tree:**  
        - **Max Depth, Min Samples Split and Min Samples Leaf:** These control the complexity of the tree.
        
        **For XGBoost only:**  
        - **Learning Rate:** Determines the speed at which the model learns; a lower rate requires more trees.
        """)
    # Model selection
    model_choice = st.sidebar.radio("Choose Model", ('Random Forest', 'XGBoost', 'Decision Tree'))

    # Test size slider
    test_size = st.sidebar.slider('Test Set Size', 0.1, 0.5, 0.2)

    st.sidebar.header('Input Wine Characteristics')
    intro_box("""
    Enter values for the wine characteristics below. These values will be used to make a prediction
    once the model is trained.
    """)
    user_input = {}
    # Assume features are all columns except 'quality' and 'color'
    for feature in data.drop(['quality', 'color'], axis=1).columns:
        user_input[feature] = st.number_input(f'{feature}',
                                              float(data[feature].min()),
                                              float(data[feature].max()),
                                              float(data[feature].mean()))

    st.sidebar.header('Model Hyperparameter Settings')
    # Depending on the chosen model, show the appropriate hyperparameter sliders
    if model_choice in ['Random Forest', 'XGBoost']:
        n_estimators = st.sidebar.slider('Number of Trees', 10, 200, 100)
        max_depth = st.sidebar.slider('Max Depth', 1, 20, 10 if model_choice=='Random Forest' else 6)
        if model_choice == 'Random Forest':
            min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2)
            min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, 10, 1)
        else:  # XGBoost
            learning_rate = st.sidebar.slider('Learning Rate', 0.01, 0.3, 0.1)
    else:  # Decision Tree
        max_depth = st.sidebar.slider('Max Depth', 1, 20, 5)
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, 10, 1)

    # Prepare features and target based on user's prediction choice
    if prediction_choice == 'Quality':
        target = 'quality'
        features = data.drop(['quality', 'color'], axis=1).columns
    else:
        target = 'color'
        features = data.drop(['quality', 'color'], axis=1).columns

    X = data[features].values
    y = data[target].values

    # Split data into training and testing sets BEFORE scaling (to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    # Fit scaler on training data only, then transform both training and test sets
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialise and train the selected model
    if model_choice == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42
        )
    elif model_choice == 'XGBoost':
        # For binary prediction (Colour), compute class imbalance ratio
        if prediction_choice == 'Colour':
            counts = np.bincount(y_train)
            # Ensure there are two classes
            if len(counts) == 2:
                scale_pos_weight = counts[0] / counts[1]
            else:
                scale_pos_weight = 1
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                random_state=42
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
    else:  # Decision Tree
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42
        )
    model.fit(X_train_scaled, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    st.write(f'## Model Performance ({model_choice} - {prediction_choice})')
    explanation_box("""
    These metrics illustrate model effectiveness. Accuracy represents the overall rate of correct predictions.
    Precision measures how often the model's positive predictions are correct, recall shows how well the model identifies
    all positive cases, and the F1-score provides a balanced metric combining precision and recall.
    """)
    st.markdown(f'<div class="result-box">Metrics:<br>Accuracy: {accuracy:.2f}<br>Precision: {precision:.2f}<br>Recall: {recall:.2f}<br>F1-score: {f1:.2f}</div>', unsafe_allow_html=True)

    # Make prediction based on user input
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction_result = model.predict(input_scaled)[0]

    # Interpret the prediction in UK English
    if prediction_choice == 'Quality':
        quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}
        predicted_result = quality_mapping_reverse.get(prediction_result, "Unknown")
    else:
        predicted_result = 'white' if prediction_result == 1 else 'red'

    st.markdown(f'<div class="result-box">Predicted {prediction_choice}: {predicted_result}</div>', unsafe_allow_html=True)

    # Plot Feature Importances (if available for tree-based models)
    if model_choice in ['Random Forest', 'Decision Tree', 'XGBoost']:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        top_features = feature_importance.head(5)
        st.write('## Top 5 Most Important Features')
        explanation_box("""
        This chart shows which wine characteristics are most important for making predictions.
        Longer bars indicate more influential features.
        """)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
        ax_imp.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        ax_imp.set_xlabel('Importance')
        ax_imp.set_ylabel('Feature')
        ax_imp.set_title(f'Top 5 Feature Importances ({model_choice})')
        ax_imp.invert_yaxis()
        st.pyplot(fig_imp)

    # Plot ROC Curve
    st.write('## ROC Curve')
    explanation_box("""
    The ROC (Receiver Operating Characteristic) curve illustrates the model's ability to distinguish between classes.
    For multi-class tasks, the curves are generated for each class; for binary tasks, a single curve is shown.
    A curve closer to the top-left corner indicates better performance, and the AUC (Area Under Curve) summarises this performance.
    """)
    if prediction_choice == 'Quality':
        # Multi-class ROC: use label binarisation
        classes_present = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes_present)
        y_prob = model.predict_proba(X_test_scaled)
        fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
        for i, class_val in enumerate(classes_present):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, lw=2, label=f'Class {class_val} (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'Multi-class ROC Curve ({model_choice})')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        # Binary ROC for Colour prediction
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_bin, ax_bin = plt.subplots(figsize=(10, 6))
        ax_bin.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax_bin.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_bin.set_xlabel('False Positive Rate')
        ax_bin.set_ylabel('True Positive Rate')
        ax_bin.set_title(f'ROC Curve ({model_choice})')
        ax_bin.legend(loc="lower right")
        st.pyplot(fig_bin)

    # Footer (using relative positioning so it does not obscure content on smaller screens)
    footer = """
    <style>
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
      <p>© 2024 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

