"""
Text preprocessing module for spam detection.

This module provides functions to clean and vectorize text data
for machine learning classification tasks.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class TextPreprocessor:
    """Handles text cleaning and vectorization."""

    def __init__(self, max_features: int = 3000, use_stemming: bool = True):
        """
        Initialize preprocessor.

        Args:
            max_features: Maximum number of TF-IDF features
            use_stemming: Whether to apply stemming
        """
        self.max_features = max_features
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True, stop_words="english")

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Raw text input

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove phone numbers (basic pattern)
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove stopwords and apply stemming
        if self.stemmer:
            words = text.split()
            words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
            text = " ".join(words)

        return text

    def fit_transform(self, texts: list) -> tuple:
        """
        Clean texts and fit TF-IDF vectorizer.

        Args:
            texts: List of text strings

        Returns:
            Tuple of (cleaned_texts, tfidf_features)
        """
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Fit and transform with TF-IDF
        features = self.vectorizer.fit_transform(cleaned_texts)

        return cleaned_texts, features

    def transform(self, texts: list):
        """
        Transform new texts using fitted vectorizer.

        Args:
            texts: List of text strings

        Returns:
            TF-IDF features
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)

    def get_feature_names(self) -> list:
        """Get TF-IDF feature names."""
        return self.vectorizer.get_feature_names_out()


def preprocess_text(text: str) -> str:
    """
    Quick preprocessing function for single text (for Streamlit app).

    Args:
        text: Raw text input

    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor(use_stemming=False)
    return preprocessor.clean_text(text)


if __name__ == "__main__":
    # Test preprocessing
    sample_texts = [
        "Win FREE iPhone now! Call 123-456-7890",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Click here: http://fake.com",
    ]

    preprocessor = TextPreprocessor()

    print("Testing text cleaning:")
    for text in sample_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned: {cleaned}")

    print("\n" + "=" * 50)
    print("Testing TF-IDF vectorization:")
    cleaned_texts, features = preprocessor.fit_transform(sample_texts)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Sample feature names: {preprocessor.get_feature_names()[:10]}")
