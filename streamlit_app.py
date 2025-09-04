import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Download NLTK resources (only runs once)
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Starbucks Reviews Sentiment Analysis (SVM)")
st.markdown("Negative = 0 | Neutral = 1 | Positive = 2")

# File upload (instead of hardcoding path)
uploaded_file = st.file_uploader("Upload your Starbucks Reviews CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Preprocessing
    # -------------------------------
    def label_sentiment(rating):
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive

    df['Sentiment'] = df['Rating'].apply(label_sentiment)

    # Clean text
    def clean_text(text):
        text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep letters/spaces only
        return text.lower()

    df['Cleaned_Review'] = df['Review'].apply(clean_text)

    # Stopwords + Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

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
    # Metrics
    # -------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    st.subheader("📈 Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])
    disp.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
    axes[0].set_title('Confusion Matrix')

    # Metrics bar chart
    metrics = {
        'Accuracy': accuracy,
        'Precision (macro)': precision,
        'Recall (macro)': recall,
        'F1 Score (macro)': f1
    }

    axes[1].bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple'])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Performance Metrics')
    axes[1].set_ylabel('Score')
    for i, v in enumerate(metrics.values()):
        axes[1].text(i, v + 0.03, f"{v:.2f}", ha='center', fontsize=12)

    st.pyplot(fig)

    # -------------------------------
    # Try Custom Review
    # -------------------------------
    st.subheader("📝 Test Your Own Review")
    user_input = st.text_area("Enter a review text:")
    if user_input:
        processed = preprocess(clean_text(user_input))
        vec = tfidf.transform([processed])
        prediction = svm.predict(vec)[0]
        sentiment_label = ['Negative', 'Neutral', 'Positive'][prediction]
        st.success(f"Predicted Sentiment: **{sentiment_label}**")

