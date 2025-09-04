import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



st.title("Starbucks Reviews Sentiment Analysis (SVM) - 3 Classes (Neg/Neu/Pos)")

stop_words = set(ENGLISH_STOP_WORDS)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Label sentiment with 3 classes: Negative=0, Neutral=2, Positive=1
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else 2))
    return df

@st.cache_data
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize each token
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@st.cache_data
def train_and_evaluate(data):
    data['Processed_Review'] = data['Review'].apply(preprocess_text)
    X = data['Processed_Review']
    y = data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    svm = LinearSVC()
    svm.fit(X_train_tfidf, y_train)
    y_pred = svm.predict(X_test_tfidf)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    return svm, metrics, vectorizer, X_test, y_test, y_pred

def predict_with_confidence(model, vectorized_input, threshold=0.5):
    decision_scores = model.decision_function(vectorized_input)
    # For binary classification this is one-dimensional; for multi-class use max margin
    if len(decision_scores.shape) == 1:
        confidence = abs(decision_scores[0])  # distance from hyperplane
    else:
        confidence = max(decision_scores[0]) - sorted(decision_scores[0])[-2]  # margin between top 2 classes
    
    if confidence < threshold:
        return 2  # Neutral label
    else:
        return model.predict(vectorized_input)[0]


# Sidebar file uploader
uploaded_file = st.file_uploader("Upload Starbucks reviews CSV file", type=['csv'])

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Raw Data Sample")
    st.dataframe(data.head())

    st.write("### Training and Evaluating SVM Model...")
    model, metrics, vectorizer, X_test, y_test, y_pred = train_and_evaluate(data)

    st.write("### Evaluation Metrics")
    st.json(metrics)

    results_df = pd.DataFrame({
        'Review': X_test,
        'Actual Sentiment': y_test,
        'Predicted Sentiment': y_pred
    }).reset_index(drop=True)

    sentiment_labels = {0: "Negative", 1: "Positive", 2: "Neutral"}
    
    results_df['Actual Sentiment'] = results_df['Actual Sentiment'].map(sentiment_labels)
    results_df['Predicted Sentiment'] = results_df['Predicted Sentiment'].map(sentiment_labels)

    st.write("### Sample Predictions")
    st.dataframe(results_df.head(20))

    # User input for live prediction
    st.write("### Try Your Own Review:")
    user_input = st.text_area("Enter a Starbucks review to predict its sentiment:")

    if user_input:
        def preprocess_single(text):
            text = re.sub(r'<.*?>', '', str(text))
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()
            tokens = text.split()
            tokens = [word for word in tokens if word not in stop_words]
            return ' '.join(tokens)

        user_processed = preprocess_single(user_input)
        user_vect = vectorizer.transform([user_processed])
        pred = predict_with_confidence(model, user_vect)
        st.write("**Predicted Sentiment:**", sentiment_labels.get(pred, "Unknown"))
        st.write("Cleaned Input:", user_processed)

else:
    st.info("Please upload the Starbucks reviews CSV file to begin.")


