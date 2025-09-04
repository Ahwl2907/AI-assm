import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
import streamlit as st

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# -------------------------------
# NLTK Downloads
# -------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
# Newer NLTK versions renamed the tagger
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Starbucks Reviews Sentiment Analysis (SVM)")
st.markdown("Negative = 0 | Neutral = 1 | Positive = 2")

# File upload
uploaded_file = st.file_uploader("Upload your Starbucks Reviews CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Sentiment labeling
    # -------------------------------
    def label_sentiment(rating):
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive

    df['Sentiment'] = df['Rating'].apply(label_sentiment)

    # -------------------------------
    # Text Cleaning
    # -------------------------------
    def clean_text(text):
        text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep letters/spaces only
        return text.lower()

    df['Cleaned_Review'] = df['Review'].apply(clean_text)

    # -------------------------------
    # Preprocessing (POS-aware with fallback)
    # -------------------------------
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        """Map POS tag to WordNet format."""
        tag = pos_tag([word], lang='eng')[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(text):
        tokens = text.split()
        processed_tokens = []
        for word in tokens:
            if word in stop_words:
                continue
            try:
                pos = get_wordnet_pos(word)  # Try POS-aware lemmatization
                lemma = lemmatizer.lemmatize(word, pos)
            except Exception:
                # Fallback if POS tagger fails
                lemma = lemmatizer.lemmatize(word)
            processed_tokens.append(lemma)
        return ' '.join(processed_tokens)

    df['Processed_Review'] = df['Cleaned_Review'].apply(preprocess)

    st.subheader("📌 Data Preview")
    st.write(df.head())

    # -------------------------------
    # Train/Test Split + TF-IDF
    # -------------------------------
    X = df['Processed_Review']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # -------------------------------
    # Train SVM Model
    # -------------------------------
    svm = LinearSVC()
    svm.fit(X_train_tfidf, y_train)
    y_pred = svm.predict(X_test_tfidf)

    # -------------------------------
    #
