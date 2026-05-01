"""Text preprocessing utilities for clustering experiments.

Keep functions in this module independent from project-specific file paths so
they can be reused from notebooks, scripts, and tests.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
import string


PUNCTUATION_TRANSLATOR = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Normalize a single text document before vectorization.

    The cleaning is intentionally simple for this assignment:
    lowercase text, remove basic punctuation, and collapse extra whitespace.
    """
    normalized = str(text).lower().translate(PUNCTUATION_TRANSLATOR).strip()
    return re.sub(r"\s+", " ", normalized)


def preprocess_corpus(documents: Iterable[str]) -> list[str]:
    """Apply text preprocessing to a collection of documents.

    Args:
        documents: Raw text documents from any source.

    Returns:
        A list of cleaned text documents ready for feature extraction.
    """
    return [clean_text(document) for document in documents]


def vectorize_text(documents: Iterable[str], **vectorizer_kwargs):
    """Convert cleaned text documents into numerical features.

    Uses TF-IDF with English stopword removal by default. Extra keyword
    arguments are forwarded to ``TfidfVectorizer`` and may override defaults.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer_options = {"stop_words": "english"}
    vectorizer_options.update(vectorizer_kwargs)

    vectorizer = TfidfVectorizer(**vectorizer_options)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def build_tfidf_features(documents: Iterable[str], **vectorizer_kwargs):
    """Run the full preprocessing pipeline and return TF-IDF features.

    Args:
        documents: Raw text documents.
        **vectorizer_kwargs: Optional settings for ``TfidfVectorizer``.

    Returns:
        A tuple containing the TF-IDF feature matrix and fitted vectorizer.
    """
    cleaned_documents = preprocess_corpus(documents)
    return vectorize_text(cleaned_documents, **vectorizer_kwargs)
