import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("Starbucks Reviews Sentiment Analysis (SVM)")

stop_words = set(ENGLISH_STOP_WORDS)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    #df = df[df['Rating'] != 3]  # Remove neutral reviews
    #df['Sentiment'] = np.where(df['Rating'] >= 4, 1, 0)  # 1=positive, 0=negative
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else 2))
    return df

@st.cache_data
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', str(text))
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    # Remove stopwords without lemmatization
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@st.cache_data
def train_and_evaluate(data):
    data['Processed_Review'] = data['Review'].apply(preprocess_text)
    X = data['Processed_Review']
    y = data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    
    st.write("### Sample Predictions")
    st.dataframe(results_df.head(20))

    # New area for user input text
st.write("### Try Your Own Review:")
user_input = st.text_area("Enter a Starbucks review to predict its sentiment:")

if user_input:
    # Preprocess input text same as training data
    def preprocess_single(text):
        import re
        text = re.sub(r'<.*?>', '', str(text))
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    user_processed = preprocess_single(user_input)
    
    # Vectorize the processed input text
    user_vect = vectorizer.transform([user_processed])
    
    # Predict sentiment: 1=Positive, 0=Negative (Neutral not used in training)
    pred = model.predict(user_vect)[0]
    
    # Map predicted label to text
    sentiment_map = {1: "Positive", 0: "Negative", 2: "Neutral"}
    
    # Display prediction
    st.write("**Predicted Sentiment:**", sentiment_map.get(pred, "Unknown"))

    # Optionally show the cleaned and processed versions too
    st.write("Cleaned Input:", user_processed)
    
else:
    st.info("Please upload the Starbucks reviews CSV file to begin.")


