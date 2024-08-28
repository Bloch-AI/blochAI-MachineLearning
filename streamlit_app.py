import streamlit as st
import pandas as pd
import requests
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb

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
    </style>
    """, unsafe_allow_html=True)

def explanation_box(text):
    return st.markdown(f'<div class="explanation-box">{text}</div>', unsafe_allow_html=True)

def intro_box(text):
    return st.markdown(f'<div class="intro-box">{text}</div>', unsafe_allow_html=True)

# Add header to the app
st.markdown('<div class="header"><h1>🍷 Wine Quality Prediction App</h1></div>', unsafe_allow_html=True)

# Intro box explaining the app
intro_box("""
This app demonstrates how machine learning works, focusing on simple classification algorithms. You can experiment 
with the data, adjust parameters and select different models to observe how these changes affect the model's performance. 
By interacting with these elements, you'll gain practical insights into the workings of machine learning systems.
""")

# GitHub URL for the wine dataset
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

# Load the dataset
data = load_data(url)
if data is None:
    st.stop()

# Display the first few rows of the dataset
st.write('###Wine Dataset')
explanation_box("""
This data, drawn from the UCI Machine Learning Repository wine dataset, has trained our machine learning model. 
It features two class columns: Quality and Colour. Using machine learning, we can predict these attributes based 
on the listed chemical components.
""")
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
    
    st.header('Model Parameters')
    
    intro_box("""
        You can adjust the options below to experiment with different model settings. 
        The results will automatically adjust. 
    """)
    
    explanation_box("""
        Choose whether to predict wine quality (how good the wine is) or 
        colour (red or white). This determines what the model will try to guess.
    """)
    prediction_choice = st.radio("Choose what to predict", ('Quality', 'Color'))
    
    explanation_box("""
When selecting a machine learning model, you have several options, each with its own approach and strengths. The Decision Tree model 
mimics a flowchart, making choices based on a series of yes/no questions about the data features. This simple structure makes Decision 
Trees easy to interpret, but they may struggle with complex patterns. Random Forest builds upon this concept by combining multiple decision 
trees, aggregating their predictions to make a final decision - imagine a forest of trees voting on the outcome. This ensemble approach 
often yields more robust results. XGBoost takes tree-based modelling further by building trees sequentially, with each new tree focusing on 
correcting the errors of the previous ones, resulting in a powerful and adaptive model that often achieves state-of-the-art performance on 
many tasks. However, the more complex a model is the longer it takes to run. 
    """)
    model_choice = st.radio("Choose model", ('Random Forest', 'XGBoost', 'Decision Tree'))
    
    explanation_box("""
        This determines how much of the data is used for testing vs training the model. 
        A larger test size means more data for evaluating the model, but this also means less data for the model to learn from. 
    """)
    test_size = st.slider('Test Size', 0.1, 0.5, 0.2)

    st.write('## Model Hyperparameters')
    explanation_box("""
These settings control how our model learns from data. The number of trees determines how many 'voters' we have making decisions; 
more trees can lead to better guesses but take longer to process. Max depth is the number of branches and sets how complex each tree's decisions can be; 
deeper trees spot intricate patterns but risk 'overfitting' - memorising the training data rather than learning general rules. 
Conversely, trees that are too shallow might 'underfit', missing important patterns. The min samples split/leaf settings determine
how many data points must be in a group before the tree makes a new split or forms a leaf. This helps prevent the model from making 
decisions based on too few examples, again balancing between overfitting and underfitting. 
For XGBoost models, the learning rate determines how quickly the model absorbs new information.
    """)
    if model_choice == 'Random Forest':
        n_estimators = st.slider('Number of trees', 10, 200, 100)
        max_depth = st.slider('Max depth', 1, 20, 10)
        min_samples_split = st.slider('Min samples split', 2, 10, 2)
        min_samples_leaf = st.slider('Min samples leaf', 1, 10, 1)
    elif model_choice == 'XGBoost':
        n_estimators = st.slider('Number of trees', 10, 200, 100)
        max_depth = st.slider('Max depth', 1, 20, 6)
        learning_rate = st.slider('Learning rate', 0.01, 0.3, 0.1)
    else:  # Decision Tree
        max_depth = st.slider('Max depth', 1, 20, 5)
        min_samples_split = st.slider('Min samples split', 2, 10, 2)
        min_samples_leaf = st.slider('Min samples leaf', 1, 10, 1)

    explanation_box("""
        You can add your own wine values below, or more likely, tweak these default values to see how the model behaves.
    """)
    st.write(f'Predict Wine {prediction_choice}')
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

X = data[features].values
y = data[target].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

# Initialize and train the selected model
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
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
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

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Display model performance metrics
st.write(f'## Model Performance ({model_choice} - {prediction_choice})')
explanation_box("""
These metrics illustrate model effectiveness. Accuracy represents the overall rate of correct predictions. Precision measures how often the model's 
positive predictions are right, while recall shows how well it finds all positive cases. The F1-score balances precision and recall, providing a single, 
comprehensive measure of performance. Together, these metrics offer a thorough assessment of our model's predictive capabilities.
""")
st.markdown(f'<div class="result-box">'
            f'Metrics:<br>'
            f'Accuracy: {accuracy:.2f}<br>'
            f'Precision: {precision:.2f}<br>'
            f'Recall: {recall:.2f}<br>'
            f'F1-score: {f1:.2f}'
            f'</div>', unsafe_allow_html=True)

# Make prediction based on user input
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# Interpret the prediction
if prediction_choice == 'Quality':
    quality_mapping_reverse = {v: k for k, v in quality_mapping.items()}
    predicted_result = quality_mapping_reverse[prediction]
else:
    predicted_result = 'white' if prediction == 1 else 'red'

# Display the prediction result
st.markdown(f'<div class="result-box"> Predicted {prediction_choice}: {predicted_result}</div>', unsafe_allow_html=True)

# Calculate and display feature importances
if model_choice in ['Random Forest', 'Decision Tree', 'XGBoost']:
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    top_features = feature_importance.head(5)

    st.write('Top 5 Most Important Features')
    explanation_box("""
        This chart shows which wine characteristics are most important for making predictions. 
        Longer bars indicate more influential features.
    """)
    plt.figure(figsize=(10, 5))
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 5 Most Important Feature Importances ({model_choice})')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

# Plot ROC curve
st.write('ROC Curve')
explanation_box("""
The ROC (Receiver Operating Characteristic) curve illustrates our model's ability to distinguish between classes. 
It plots the true positive rate against the false positive rate at various classification thresholds. A curve closer to the 
top-left corner indicates better performance, as it represents a higher true positive rate and a lower false positive rate. 
The AUC (Area Under Curve) summarises the ROC curve's information into a single number between 0 and 1, with higher values 
indicating better overall classification performance.
""")
if prediction_choice == 'Quality':
    # Multi-class ROC curve for Quality prediction
    y_prob = model.predict_proba(X_test)
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
    y_prob = model.predict_proba(X_test)[:, 1]
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
