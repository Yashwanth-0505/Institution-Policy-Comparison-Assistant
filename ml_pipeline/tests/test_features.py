"""
test_features.py — Tests for ml_pipeline.features
"""
from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.features import FeatureExtractor
from ml_pipeline.models import ClauseFeatures
from ml_pipeline.tests.conftest import make_clause


def _make_clauses(n: int = 3) -> list:
    return [
        make_clause(
            clause_id=f"doc_a_clause_{i:04d}",
            text=f"Clause {i}: Employees must comply with policy number {i} at all times.",
        )
        for i in range(n)
    ]


def test_tfidf_vectors_shape() -> None:
    """TF-IDF matrix has shape (n_clauses, n_features)."""
    clauses = _make_clauses(4)
    extractor = FeatureExtractor()
    extractor.fit_tfidf(clauses)
    matrix = extractor.get_tfidf_matrix(clauses)
    assert matrix.shape[0] == 4
    assert matrix.shape[1] > 0


def test_extract_returns_clause_features() -> None:
    """extract() returns one ClauseFeatures per clause."""
    clauses = _make_clauses(3)
    extractor = FeatureExtractor()
    features = extractor.extract(clauses)
    assert len(features) == 3
    for f in features:
        assert isinstance(f, ClauseFeatures)
        assert f.tfidf_vector is not None
        assert f.tfidf_vector.shape[0] > 0


def test_embedding_failure_graceful(mocker) -> None:
    """When SentenceTransformer.encode raises, embeddings are None (no crash)."""
    # Clear class-level cache to force a fresh load attempt
    FeatureExtractor._model_cache.clear()

    # Mock SentenceTransformer to raise on encode
    mock_st = mocker.MagicMock()
    mock_st.encode.side_effect = RuntimeError("Mock embedding failure")
    mocker.patch(
        "ml_pipeline.features._SentenceTransformer",
        return_value=mock_st,
    )
    mocker.patch("ml_pipeline.features._ST_AVAILABLE", True)

    # Clear cache again after patching
    FeatureExtractor._model_cache.clear()

    extractor = FeatureExtractor()
    # Force the embedding model to be the mock
    extractor._embedding_model = mock_st

    clauses = _make_clauses(2)
    embeddings = extractor.get_embeddings(clauses)
    assert embeddings is None


def test_same_clause_same_tfidf() -> None:
    """The same clause text always produces the same TF-IDF vector."""
    text = "All employees must complete mandatory training within 30 days of joining."
    c1 = make_clause(clause_id="doc_a_clause_0000", text=text)
    c2 = make_clause(clause_id="doc_a_clause_0001", text=text)
    extractor = FeatureExtractor()
    extractor.fit_tfidf([c1, c2])
    matrix = extractor.get_tfidf_matrix([c1, c2])
    np.testing.assert_array_almost_equal(matrix[0], matrix[1])
