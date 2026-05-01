"""Clustering helpers for text mining experiments."""

from __future__ import annotations

import numpy as np


def _to_dense(feature_matrix):
    """Return a dense array when an estimator does not accept sparse input."""
    if hasattr(feature_matrix, "toarray"):
        return feature_matrix.toarray()
    return feature_matrix


def run_kmeans(feature_matrix, n_clusters: int = 5, random_state: int = 42, **kwargs):
    """Fit K-Means on a feature matrix and return the trained model.

    Args:
        feature_matrix: Numerical document features, such as a TF-IDF matrix.
        n_clusters: Number of clusters to estimate.
        random_state: Seed for reproducible experiments.
        **kwargs: Additional arguments passed to ``sklearn.cluster.KMeans``.
    """
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto", **kwargs)
    return model.fit(feature_matrix)


def run_agglomerative(feature_matrix, n_clusters: int = 5, **kwargs):
    """Fit Agglomerative Clustering and return the trained model.

    Agglomerative Clustering in scikit-learn expects dense input, so sparse
    TF-IDF matrices are converted before fitting.
    """
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    return model.fit(_to_dense(feature_matrix))


def get_cluster_labels(model, feature_matrix=None):
    """Return cluster assignments from a fitted clustering model.

    If the model already exposes ``labels_``, those labels are returned.
    Otherwise, ``feature_matrix`` is used with the model's ``predict`` method.
    """
    if hasattr(model, "labels_"):
        return model.labels_
    if feature_matrix is None:
        raise ValueError("feature_matrix is required when the model has no labels_.")
    return model.predict(feature_matrix)


def evaluate_labels(feature_matrix, labels) -> float:
    """Compute silhouette score for clustering labels.

    Returns ``nan`` when there are too few labels for silhouette scoring.
    """
    from sklearn.metrics import silhouette_score

    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return float("nan")
    return float(silhouette_score(feature_matrix, labels))


def evaluate_clustering(feature_matrix, labels) -> dict[str, float]:
    """Compute simple internal clustering metrics for experiment comparison."""
    return {"silhouette_score": evaluate_labels(feature_matrix, labels)}


def get_top_terms_per_cluster(kmeans_model, vectorizer, top_n: int = 10) -> dict[int, list[str]]:
    """Return the highest-weighted TF-IDF terms for each K-Means cluster."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_terms = {}

    for cluster_id, center in enumerate(kmeans_model.cluster_centers_):
        top_indices = center.argsort()[::-1][:top_n]
        top_terms[cluster_id] = feature_names[top_indices].tolist()

    return top_terms


def find_representative_documents(
    feature_matrix,
    labels,
    document_ids=None,
    top_n: int = 3,
) -> dict[int, list]:
    """Find documents closest to each cluster centroid.

    Args:
        feature_matrix: TF-IDF feature matrix.
        labels: Cluster label for each document.
        document_ids: Optional IDs to return instead of row positions.
        top_n: Number of representative documents per cluster.

    Returns:
        Mapping from cluster label to representative document IDs or row
        positions.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    labels = np.asarray(labels)
    ids = np.arange(len(labels)) if document_ids is None else np.asarray(document_ids)
    representatives = {}

    for cluster_id in sorted(set(labels)):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_matrix = feature_matrix[cluster_indices]
        centroid = np.asarray(cluster_matrix.mean(axis=0)).reshape(1, -1)
        similarities = cosine_similarity(cluster_matrix, centroid).ravel()
        best_local_indices = similarities.argsort()[::-1][:top_n]
        representatives[int(cluster_id)] = ids[cluster_indices[best_local_indices]].tolist()

    return representatives
