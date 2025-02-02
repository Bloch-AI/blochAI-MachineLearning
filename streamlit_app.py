#**********************************************
# Wine Classification App
# Version 1
# 1st September 2024 (Updated: 2nd February 2025)
# Jamie Crossman-Smith
# jamie@bloch.ai
#**********************************************
# This Python code creates an interactive educational web application using Streamlit.
# The application demonstrates how machine learning can be used for wine classification
# using various models. Users can experiment with the data, adjust parameters and
# select different models (Random Forest, XGBoost, Decision Tree) to observe how these changes
# affect the model's performance.
#
# The app displays:
# - The wine dataset (with a brief explanation of its columns)
# - Model performance metrics such as accuracy, precision, recall, F1-score and ROC curves
# - Feature importance charts (for tree-based models)
#
# Educational explanations are provided throughout the app (via sidebar expanders and info boxes)
# to help users understand the model parameters, hyperparameters and how to interpret the visual outputs.
#
# This app is intended to enhance understanding of machine learning in a practical, interactive manner.
#**********************************************

# Import necessary libraries for building the Streamlit app and handling data
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the wine dataset from scikit-learn and other ML libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import xgboost as xgb

# Custom CSS for styling the app (updated to match the new SVM app style in UK English)
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
    /* New explanation boxes with updated colours */
    .explanation-box {
        background-color: #d1e7dd;  /* light green */
        border: 2px solid #0f5132;  /* dark green */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .intro-box {
        background-color: #cff4fc;  /* light blue */
        border: 2px solid #055160;  /* dark blue */
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
        position: fixed;
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

def main():
    # Header & Introduction
    st.markdown('<div class="header"><h1>🍷 Wine Classification App</h1></div>', unsafe_allow_html=True)
    intro_box("""
    This app demonstrates how machine learning works by classifying wines.
    Experiment with the data, adjust parameters and select different models to observe how these changes
    affect model performance. By interacting with these elements, you'll gain practical insights into the workings
    of machine learning systems.
    """)

    # Load the wine dataset from scikit-learn
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target

    st.write('## Wine Dataset')
    explanation_box(f"""
    This dataset, available in scikit-learn, contains chemical attributes of wines along with a target
    variable representing wine classes. The target classes are:
    - **0:** {wine.target_names[0]}
    - **1:** {wine.target_names[1]}
    - **2:** {wine.target_names[2]}
    """)
    st.dataframe(df.head(), height=150)

    # Sidebar: Model Parameters & Hyperparameters
    st.sidebar.header('Model Parameters')
    with st.sidebar.expander("Learn about Model Parameters"):
        st.markdown("""
        **Prediction Target:**  
        The app predicts the wine class. Each class represents a different type of wine.
        
        **Model Choice:**  
        Select from three models:
        - **Random Forest:** An ensemble of decision trees for robust predictions.
        - **XGBoost:** An advanced boosting technique that builds trees sequentially.
        - **Decision Tree:** A simple tree-based model for clear, interpretable decisions.
        """)
    # With only one target, we simply set it here
    target = 'target'

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
    # All feature columns are numeric
    feature_cols = wine.feature_names
    user_input = {}
    for feature in feature_cols:
        user_input[feature] = st.number_input(
            f'{feature}',
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )

    st.sidebar.header('Model Hyperparameter Settings')
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

    # Prepare features and target for modelling
    X = df[feature_cols].values
    y = df[target].values

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
        # For wine, there are three classes. For multi-class, we do not set scale_pos_weight.
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

    st.write(f'## Model Performance ({model_choice})')
    explanation_box("""
    These metrics illustrate model effectiveness. Accuracy represents the overall rate of correct predictions.
    Precision measures how often the model's positive predictions are correct, recall shows how well the model identifies
    all instances of each class, and the F1-score provides a balanced metric combining precision and recall.
    """)
    st.markdown(f'<div class="result-box">Metrics:<br>Accuracy: {accuracy:.2f}<br>Precision: {precision:.2f}<br>Recall: {recall:.2f}<br>F1-score: {f1:.2f}</div>', unsafe_allow_html=True)

    # Make prediction based on user input
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction_result = model.predict(input_scaled)[0]
    predicted_result = wine.target_names[prediction_result]

    st.markdown(f'<div class="result-box">Predicted Wine Class: {predicted_result}</div>', unsafe_allow_html=True)

    # Plot Feature Importances (if available for tree-based models)
    if model_choice in ['Random Forest', 'Decision Tree', 'XGBoost']:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values(by='Importance', ascending=False)
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
    For multi-class tasks, individual curves are generated for each class; for binary tasks, a single curve is shown.
    A curve closer to the top-left corner indicates better performance, and the AUC (Area Under the Curve) summarises this performance.
    """)
    classes_present = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes_present)
    y_prob = model.predict_proba(X_test_scaled)
    fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
    for i, class_val in enumerate(classes_present):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f'Class {wine.target_names[class_val]} (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'Multi-class ROC Curve ({model_choice})')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Footer (fixed across the bottom)
    footer = """
    <style>
    .footer {
        position: fixed;
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
      <p>© 2025 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
