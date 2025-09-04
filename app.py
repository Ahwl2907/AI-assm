import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer
from scipy.special import softmax

# NLTK downloads (for lemmatizer)
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("Starbucks Reviews Sentiment Analysis (SVM) - 3 Classes")

stop_words = set(ENGLISH_STOP_WORDS)
lemmatizer = WordNetLemmatizer()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Review'])
    # Label sentiment: 0=Negative, 2=Neutral, 1=Positive
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else 2))
    return df

@st.cache_data
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
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
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
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

# Predict with confidence for all classes
def predict_with_confidences(model, vectorizer, text):
    processed = preprocess_text(text)
    vect = vectorizer.transform([processed])
    decision_scores = model.decision_function(vect)[0]  # shape (3,)
    probs = softmax(decision_scores)  # turn into pseudo-probabilities
    return processed, probs

uploaded_file = st.file_uploader("Upload Starbucks reviews CSV file", type=['csv'])

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Raw Data Sample")
    st.dataframe(data.head())

    st.write("### Training and Evaluating SVM Model...")
    model, metrics, vectorizer, X_test, y_test, y_pred = train_and_evaluate(data)

    st.write("### Evaluation Metrics")
    st.json(metrics)

    sentiment_labels = {0: "Negative", 1: "Positive", 2: "Neutral"}

    results_df = pd.DataFrame({
        'Review': X_test.reset_index(drop=True),
        'Actual Sentiment': y_test.reset_index(drop=True).map(sentiment_labels),
        'Predicted Sentiment': pd.Series(y_pred).map(sentiment_labels)
    })

    st.write("### Sample Predictions")
    st.dataframe(results_df.head(20))

    st.write("### Try Your Own Review:")
    user_input = st.text_area("Enter a Starbucks review to predict its sentiment:")

    if user_input:
        processed, probs = predict_with_confidences(model, vectorizer, user_input)
        pred_label = sentiment_labels[np.argmax(probs)]

        st.write("**Predicted Sentiment:**", pred_label)
        st.write("Cleaned Input:", processed)

        st.write("### Confidence Scores")
        conf_df = pd.DataFrame({
            "Sentiment": [sentiment_labels[i] for i in range(len(probs))],
            "Confidence": [f"{p*100:.2f}%" for p in probs]
        })
        st.dataframe(conf_df)

else:
    st.info("Please upload the Starbucks reviews CSV file to begin.")
