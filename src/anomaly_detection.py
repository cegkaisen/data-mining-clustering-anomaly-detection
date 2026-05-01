"""Anomaly detection helpers for identifying unusual text documents."""

from __future__ import annotations

from collections.abc import Iterable
import re
import string

import numpy as np
import pandas as pd


PUNCTUATION = set(string.punctuation)

LISTING_KEYWORDS = [
    "listing_id",
    "listing",
    "marketplace",
    "seller",
    "pickup",
    "collection",
    "available",
    "contact",
    "sku",
    "token",
    "coupon",
    "promo",
    "price_ref",
    "seller_code",
]

SALE_KEYWORDS = [
    "sale",
    "sell",
    "selling",
    "price",
    "asking",
    "offer",
    "discount",
    "cash",
    "deal",
    "buy",
    "bonus",
    "voucher",
]

COMMERCIAL_PHRASES = [
    "for sale",
    "available immediately",
    "charger included",
    "battery included",
    "can be tested",
    "serious offers",
    "no major defects",
    "cash on pickup",
    "pickup preferred",
    "pickup only",
    "selling because",
    "original packaging",
    "minor signs of use",
    "works perfectly",
    "well maintained",
]


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a percentage ratio, protecting against empty text."""
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def _simple_words(text: str) -> list[str]:
    """Return simple word tokens for structural feature counts."""
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def _count_patterns(text: str, patterns: list[str]) -> int:
    """Count simple keyword or phrase occurrences in lowercase text."""
    return sum(text.count(pattern) for pattern in patterns)


def compute_structural_features(documents: Iterable[str]) -> pd.DataFrame:
    """Compute simple structural text features for anomaly detection."""
    rows = []

    for document in documents:
        text = "" if pd.isna(document) else str(document)
        lowered_text = text.lower()
        length = len(text)
        words = _simple_words(text)
        word_lengths = [len(word) for word in words]

        digit_count = sum(char.isdigit() for char in text)
        punctuation_count = sum(char in PUNCTUATION for char in text)
        uppercase_count = sum(char.isupper() for char in text)
        non_alpha_count = sum(not char.isalpha() for char in text)
        repeated_char_count = len(re.findall(r"(.)\1{3,}", text))

        rows.append(
            {
                "text_length": length,
                "word_count": len(words),
                "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
                "digit_ratio": _safe_ratio(digit_count, length),
                "punctuation_ratio": _safe_ratio(punctuation_count, length),
                "uppercase_ratio": _safe_ratio(uppercase_count, length),
                "non_alpha_ratio": _safe_ratio(non_alpha_count, length),
                "repeated_char_count": repeated_char_count,
                "contains_listing_id": int("listing_id" in lowered_text),
                "listing_keyword_count": _count_patterns(lowered_text, LISTING_KEYWORDS),
                "sale_keyword_count": _count_patterns(lowered_text, SALE_KEYWORDS),
                "commercial_phrase_count": _count_patterns(lowered_text, COMMERCIAL_PHRASES),
            }
        )

    return pd.DataFrame(rows)


def reduce_tfidf_features(tfidf_matrix, n_components: int = 50, random_state: int = 42):
    """Reduce TF-IDF features with TruncatedSVD for anomaly models."""
    from sklearn.decomposition import TruncatedSVD

    max_components = min(tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    if max_components < 1:
        raise ValueError("TF-IDF matrix is too small for TruncatedSVD.")

    svd = TruncatedSVD(
        n_components=min(n_components, max_components),
        random_state=random_state,
    )
    reduced_features = svd.fit_transform(tfidf_matrix)
    return reduced_features, svd


def build_anomaly_feature_matrix(structural_features: pd.DataFrame, reduced_tfidf_features):
    """Combine scaled structural features and reduced TF-IDF features."""
    from sklearn.preprocessing import StandardScaler

    combined_features = np.hstack([structural_features.to_numpy(), reduced_tfidf_features])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    return scaled_features, scaler


def compute_listing_pattern_score(structural_features: pd.DataFrame) -> np.ndarray:
    """Compute a simple score for listing-style commercial patterns."""
    return (
        5 * structural_features["contains_listing_id"].to_numpy()
        + structural_features["listing_keyword_count"].to_numpy()
        + structural_features["sale_keyword_count"].to_numpy()
        + 2 * structural_features["commercial_phrase_count"].to_numpy()
    )


def run_isolation_forest(feature_matrix, contamination: float | str = "auto", random_state: int = 42, **kwargs):
    """Fit an Isolation Forest model on document features.

    Args:
        feature_matrix: Numerical document features.
        contamination: Expected proportion of anomalies or ``"auto"``.
        random_state: Seed for reproducible experiments.
        **kwargs: Additional arguments passed to ``IsolationForest``.
    """
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        **kwargs,
    )
    return model.fit(feature_matrix)


def run_local_outlier_factor(feature_matrix, n_neighbors: int = 20, contamination: float | str = "auto", **kwargs):
    """Fit Local Outlier Factor and return the trained model."""
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        **kwargs,
    )
    model.fit_predict(feature_matrix)
    return model


def get_anomaly_scores(model, feature_matrix):
    """Return anomaly scores for each document.

    Higher returned values indicate more anomalous observations by negating
    scikit-learn's ``score_samples`` output.
    """
    return -model.score_samples(feature_matrix)


def get_lof_scores(model):
    """Return Local Outlier Factor anomaly scores.

    Higher returned values indicate more anomalous observations.
    """
    return -model.negative_outlier_factor_


def normalize_scores(scores) -> np.ndarray:
    """Min-max normalize anomaly scores to the 0-1 range."""
    scores = np.asarray(scores, dtype=float)
    score_range = scores.max() - scores.min()
    if score_range == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / score_range


def combine_scores(*score_arrays) -> np.ndarray:
    """Combine normalized anomaly scores with a simple average."""
    normalized_scores = [normalize_scores(scores) for scores in score_arrays]
    return np.mean(normalized_scores, axis=0)


def combine_weighted_scores(score_arrays, weights) -> np.ndarray:
    """Combine normalized scores with explicit, easy-to-read weights."""
    normalized_scores = [normalize_scores(scores) for scores in score_arrays]
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.average(normalized_scores, axis=0, weights=weights)


def select_top_anomalies(document_ids, combined_scores, n_anomalies: int = 50) -> pd.DataFrame:
    """Select the highest-scoring anomalous document IDs."""
    ranking = pd.DataFrame(
        {
            "doc_id": document_ids,
            "combined_anomaly_score": combined_scores,
        }
    ).sort_values("combined_anomaly_score", ascending=False)

    return ranking.head(n_anomalies).reset_index(drop=True)


def flag_anomalies(model, feature_matrix):
    """Return boolean anomaly flags from a fitted detector.

    Scikit-learn anomaly estimators usually return ``-1`` for anomalies and
    ``1`` for inliers.
    """
    predictions = model.predict(feature_matrix)
    return predictions == -1
