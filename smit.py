import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set page config
st.set_page_config(page_title="Cancer Data Analysis Dashboard", layout="wide")

# Title
st.title("Breast Cancer Data Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Cancer_Data.csv')
    # Clean data
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
    return df

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
options = st.sidebar.radio("Choose a section:", ["Data Overview", "Data Cleaning", "Exploratory Data Analysis", "Machine Learning"])

if options == "Data Overview":
    st.header("Data Overview")
    st.write("Shape of the dataset:", df.shape)
    st.write("First 5 rows:")
    st.dataframe(df.head())
    st.write("Data types and non-null counts:")
    st.dataframe(df.info())

elif options == "Data Cleaning":
    st.header("Data Cleaning")
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    
    st.subheader("Duplicate Rows")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        st.write("Duplicates removed.")
    
    # Drop 'id' and 'Unnamed: 32' columns as they are not useful
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        st.write("'id' column dropped.")
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
        st.write("'Unnamed: 32' column dropped.")
    
    st.subheader("Cleaned Data Shape")
    st.write(df.shape)

elif options == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # Diagnosis distribution
    st.subheader("Diagnosis Distribution")
    fig = px.pie(df, names='diagnosis', title='Diagnosis Distribution')
    st.plotly_chart(fig)
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[float, int])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Histograms
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select a feature to plot histogram:", numeric_df.columns)
    fig = px.histogram(df, x=feature, color='diagnosis', title=f'Histogram of {feature}')
    st.plotly_chart(fig)
    
    # Box plots
    st.subheader("Box Plots")
    fig = px.box(df, x='diagnosis', y=feature, title=f'Box Plot of {feature} by Diagnosis')
    st.plotly_chart(fig)

elif options == "Machine Learning":
    st.header("Machine Learning Model")
    
    # Prepare data
    df_ml = df.copy()
    df_ml['diagnosis'] = df_ml['diagnosis'].map({'M': 1, 'B': 0})
    
    X = df_ml.drop('diagnosis', axis=1)
    y = df_ml['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    fig = px.bar(importance.head(10), x='importance', y='feature', orientation='h', title='Top 10 Feature Importances')
    st.plotly_chart(fig)
