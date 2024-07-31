import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

# Ensure NLTK resources are available
nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load('./artifacts/model.pkl')
vectorizer = joblib.load('./artifacts/vectorizer.pkl')  # Assuming you saved the vectorizer

def preprocess_text(text):
    """Preprocess the input text."""
    text = text.lower()
    text = re.sub(r'\b[\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', 'email', text)
    text = re.sub(r'http[s]?://\S+', 'webaddress', text)
    text = re.sub(r'[£$]', 'moneysymb', text)
    text = re.sub(r'\b\d{10}\b', 'phonenumber', text)
    text = re.sub(r'\d+', 'numbr', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stopwords)
    
    return text

def classify_email(text):
    """Classify the input text as spam or not spam."""
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Vectorize the text
    text_vector = vectorizer.transform([processed_text])
    # Predict
    prediction = model.predict(text_vector)
    print(prediction)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

