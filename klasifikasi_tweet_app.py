
import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Title
st.title("Klasifikasi Emosi dari Tweet")

# Upload CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Awal:")
    st.write(df.head())

    # Cek missing value
    st.subheader("Info Data:")
    st.write(df.info())
    st.write("Missing Value:")
    st.write(df.isnull().sum())

    # Preprocessing text
    def clean_text(text):
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?:\/\/\S+', '', text)
        text = re.sub(r'\n', ' ', text)
        return text

    df['clean_text'] = df['Tweets'].astype(str).apply(clean_text)

    # Split data
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pilihan model
    st.subheader("Pilih Model Klasifikasi")
    model_choice = st.selectbox("Model", ["SVM", "Random Forest"])

    if model_choice == "SVM":
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC())
        ])
    else:
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier())
        ])

    # Training
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=False)

    st.subheader("Hasil Evaluasi")
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.text("Classification Report:")
    st.text(report)
