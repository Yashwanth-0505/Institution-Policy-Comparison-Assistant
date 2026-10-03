"""
features.py — Feature extraction for the POLICYX ML pipeline.

Computes TF-IDF vectors and sentence embeddings for extracted clauses.
The SentenceTransformer model is loaded once as a class attribute to avoid
repeated expensive reloads inside per-clause loops.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_pipeline.config import PipelineConfig, load_config
from ml_pipeline.models import ClauseFeatures, ExtractedClause

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SentenceTransformer — imported lazily so the module still loads
# in environments where sentence-transformers is not installed.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SentenceTransformer = None  # type: ignore[assignment,misc]
    _ST_AVAILABLE = False


class FeatureExtractor:
    """
    Extracts TF-IDF and sentence-embedding features for a list of clauses.

    The underlying :class:`~sentence_transformers.SentenceTransformer` model
    is stored as a **class-level** cache keyed by model name so that it is
    loaded only once per Python process regardless of how many
    ``FeatureExtractor`` instances are created.

    Parameters
    ----------
    config:
        Pipeline configuration.  If *None*, defaults are loaded from the
        environment.
    """

    # Class-level model cache: {model_name: SentenceTransformer | None}
    _model_cache: dict[str, Optional[object]] = {}

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config: PipelineConfig = config or load_config()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._embedding_model: Optional[object] = self._load_embedding_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_embedding_model(self) -> Optional[object]:
        """
        Return the cached SentenceTransformer, loading it if necessary.

        Returns *None* gracefully when the library is unavailable or the
        model cannot be loaded (e.g. no internet at evaluation time).
        """
        if not _ST_AVAILABLE:
            logger.warning(
                "sentence-transformers not installed — embeddings disabled."
            )
            return None

        model_name = self.config.embedding_model
        if model_name in FeatureExtractor._model_cache:
            return FeatureExtractor._model_cache[model_name]

        try:
            model = _SentenceTransformer(model_name)  # type: ignore[misc]
            FeatureExtractor._model_cache[model_name] = model
            logger.info("Loaded SentenceTransformer model: %s", model_name)
            return model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load SentenceTransformer '%s': %s — "
                "embeddings disabled.",
                model_name,
                exc,
            )
            FeatureExtractor._model_cache[model_name] = None
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_tfidf(self, clauses: list[ExtractedClause]) -> None:
        """
        Fit a TF-IDF vectoriser on the normalised text of *clauses*.

        Uses bigrams (``ngram_range=(1, 2)``), no minimum document frequency
        filter, and a 5 000-feature vocabulary cap.

        Parameters
        ----------
        clauses:
            The full corpus to fit on.  Must contain at least one clause.
        """
        if not clauses:
            raise ValueError("Cannot fit TF-IDF on an empty clause list.")

        texts = [c.normalized_text for c in clauses]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
        )
        self._vectorizer.fit(texts)
        logger.debug("TF-IDF vectoriser fitted on %d clauses.", len(clauses))

    def get_tfidf_matrix(self, clauses: list[ExtractedClause]) -> np.ndarray:
        """
        Transform *clauses* into a TF-IDF matrix.

        Parameters
        ----------
        clauses:
            Clauses to transform.  The vectoriser must have been fitted first
            via :meth:`fit_tfidf`.

        Returns
        -------
        np.ndarray
            Dense matrix of shape ``(len(clauses), n_features)``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit_tfidf`.
        """
        if self._vectorizer is None:
            raise RuntimeError(
                "TF-IDF vectoriser has not been fitted. "
                "Call fit_tfidf() before get_tfidf_matrix()."
            )
        texts = [c.normalized_text for c in clauses]
        sparse = self._vectorizer.transform(texts)
        return sparse.toarray()  # type: ignore[return-value]

    def get_embeddings(self, clauses: list[ExtractedClause]) -> Optional[np.ndarray]:
        """
        Compute sentence embeddings for *clauses*.

        Parameters
        ----------
        clauses:
            Clauses to embed.

        Returns
        -------
        np.ndarray | None
            Matrix of shape ``(len(clauses), embedding_dim)``, or *None* when
            the embedding model is unavailable.
        """
        if self._embedding_model is None:
            return None

        texts = [c.normalized_text for c in clauses]
        try:
            embeddings = self._embedding_model.encode(  # type: ignore[attr-defined]
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.array(embeddings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding generation failed: %s — falling back.", exc)
            return None

    def extract(self, clauses: list[ExtractedClause]) -> list[ClauseFeatures]:
        """
        Fit the vectoriser (if needed) and return per-clause feature objects.

        This is the primary entry point for callers that want all features in
        one call.  It fits TF-IDF on *clauses*, transforms them, and encodes
        embeddings when available.

        Parameters
        ----------
        clauses:
            The clauses to featurise.

        Returns
        -------
        list[ClauseFeatures]
            One :class:`~ml_pipeline.models.ClauseFeatures` per clause.
        """
        if not clauses:
            return []

        self.fit_tfidf(clauses)
        tfidf_matrix = self.get_tfidf_matrix(clauses)
        embeddings = self.get_embeddings(clauses)

        features: list[ClauseFeatures] = []
        for i, clause in enumerate(clauses):
            emb = embeddings[i] if embeddings is not None else None
            features.append(
                ClauseFeatures(
                    clause=clause,
                    tfidf_vector=tfidf_matrix[i],
                    embedding=emb,
                )
            )
        return features
