"""
aligner.py — Clause alignment/matching for the POLICYX ML pipeline.

Provides two matchers:
- :class:`BaselineMatcher`  — TF-IDF cosine + linear assignment (Hungarian).
- :class:`HybridMatcher`    — Weighted combination of clause-number, heading,
                              and embedding scores with review flagging.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from ml_pipeline.config import PipelineConfig, load_config
from ml_pipeline.models import ClauseFeatures, ClauseMatch, ExtractedClause, MatchType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Returns 0.0 for zero-magnitude vectors to avoid division by zero.

    Parameters
    ----------
    a, b:
        1-D numpy arrays of equal length.

    Returns
    -------
    float
        Value in ``[-1.0, 1.0]``; clipped to ``[0.0, 1.0]`` for use as a
        score.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Clip to [0, 1] — negative cosine means orthogonal/opposite, treat as 0.
    return max(0.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Baseline Matcher
# ---------------------------------------------------------------------------

class BaselineMatcher:
    """
    Aligns two clause lists using TF-IDF cosine similarity and the Hungarian
    algorithm (``scipy.optimize.linear_sum_assignment``).

    Unmatched clauses from document A become :attr:`~MatchType.REMOVED_CANDIDATE`
    entries; unmatched clauses from document B become
    :attr:`~MatchType.ADDED_CANDIDATE` entries.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or load_config()

    def match(
        self,
        features_a: list[ClauseFeatures],
        features_b: list[ClauseFeatures],
    ) -> list[ClauseMatch]:
        """
        Produce a one-to-one mapping between clauses in *features_a* and
        *features_b* based on TF-IDF cosine similarity.

        Parameters
        ----------
        features_a, features_b:
            Feature lists for the two documents.

        Returns
        -------
        list[ClauseMatch]
            Matched, added, and removed clause pairs.
        """
        if not features_a and not features_b:
            return []

        if not features_a:
            return [
                ClauseMatch(
                    clause_a=f.clause,
                    clause_b=None,
                    match_type=MatchType.ADDED_CANDIDATE,
                    tfidf_score=0.0,
                )
                for f in features_b
            ]

        if not features_b:
            return [
                ClauseMatch(
                    clause_a=f.clause,
                    clause_b=None,
                    match_type=MatchType.REMOVED_CANDIDATE,
                    tfidf_score=0.0,
                )
                for f in features_a
            ]

        tfidf_a = np.vstack([f.tfidf_vector for f in features_a])
        tfidf_b = np.vstack([f.tfidf_vector for f in features_b])

        # Build similarity matrix (norms pre-computed for efficiency).
        norms_a = np.linalg.norm(tfidf_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(tfidf_b, axis=1, keepdims=True)
        # Avoid division by zero.
        norms_a = np.where(norms_a == 0, 1e-10, norms_a)
        norms_b = np.where(norms_b == 0, 1e-10, norms_b)

        sim_matrix = (tfidf_a / norms_a) @ (tfidf_b / norms_b).T
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

        # Hungarian algorithm maximises sum — negate for minimisation.
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)

        matched_a: set[int] = set()
        matched_b: set[int] = set()
        results: list[ClauseMatch] = []

        for r, c in zip(row_ind, col_ind):
            score = float(sim_matrix[r, c])
            results.append(
                ClauseMatch(
                    clause_a=features_a[r].clause,
                    clause_b=features_b[c].clause,
                    match_score=score,
                    tfidf_score=score,
                    match_type=MatchType.BASELINE,
                )
            )
            matched_a.add(r)
            matched_b.add(c)

        # Unmatched clauses from A → removed candidates.
        for i, fa in enumerate(features_a):
            if i not in matched_a:
                results.append(
                    ClauseMatch(
                        clause_a=fa.clause,
                        clause_b=None,
                        match_type=MatchType.REMOVED_CANDIDATE,
                    )
                )

        # Unmatched clauses from B → added candidates.
        for j, fb in enumerate(features_b):
            if j not in matched_b:
                results.append(
                    ClauseMatch(
                        clause_a=fb.clause,  # repurpose clause_a slot for added
                        clause_b=None,
                        match_type=MatchType.ADDED_CANDIDATE,
                    )
                )

        return results


# ---------------------------------------------------------------------------
# Hybrid Matcher
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^([\d]+(?:\.[\d]+)*)")


def _extract_clause_number(clause: ExtractedClause) -> Optional[str]:
    """Extract the leading clause number from a clause heading, if any."""
    heading = clause.heading.strip()
    m = _NUMBER_RE.match(heading)
    return m.group(1) if m else None


class HybridMatcher:
    """
    Multi-signal clause aligner combining clause-number, heading-text, and
    semantic-embedding scores with configurable weights.

    Score formula::

        S = number_weight * N + heading_weight * H + embedding_weight * E

    where ``E`` falls back to TF-IDF cosine when embeddings are unavailable.

    Matches with ``S >= accept_threshold`` are accepted as
    :attr:`~MatchType.HYBRID`.  Matches in ``[review_threshold, accept_threshold)``
    are flagged as :attr:`~MatchType.NEEDS_REVIEW`.  Lower-scoring pairs are
    left unmatched.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or load_config()

    # ------------------------------------------------------------------
    # Score sub-components
    # ------------------------------------------------------------------

    def _number_score(self, clause_a: ExtractedClause, clause_b: ExtractedClause) -> float:
        """
        Clause-number similarity score.

        Returns
        -------
        float
            1.0 for identical numbers, 0.5 for shared prefix, 0.0 otherwise.
        """
        num_a = _extract_clause_number(clause_a)
        num_b = _extract_clause_number(clause_b)
        if num_a is None or num_b is None:
            return 0.0
        if num_a == num_b:
            return 1.0
        # Prefix match (e.g. "3.1" vs "3.1.2")
        shorter = min(num_a, num_b, key=len)
        longer = max(num_a, num_b, key=len)
        if longer.startswith(shorter):
            return 0.5
        return 0.0

    def _heading_score(self, clause_a: ExtractedClause, clause_b: ExtractedClause) -> float:
        """
        Heading text similarity using RapidFuzz token-sort ratio.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.
        """
        h_a = clause_a.heading.strip()
        h_b = clause_b.heading.strip()
        if not h_a and not h_b:
            return 0.0
        ratio = fuzz.token_sort_ratio(h_a, h_b)
        return ratio / 100.0

    def _embedding_score(
        self,
        feat_a: ClauseFeatures,
        feat_b: ClauseFeatures,
    ) -> float:
        """
        Semantic similarity score; falls back to TF-IDF cosine when embeddings
        are unavailable.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.
        """
        if feat_a.embedding is not None and feat_b.embedding is not None:
            return cosine_sim(
                np.asarray(feat_a.embedding),
                np.asarray(feat_b.embedding),
            )
        # Fallback: TF-IDF cosine.
        return cosine_sim(
            np.asarray(feat_a.tfidf_vector),
            np.asarray(feat_b.tfidf_vector),
        )

    def _combined_score(self, n: float, h: float, e: float) -> float:
        """
        Compute the weighted combined score.

        Parameters
        ----------
        n: clause-number score
        h: heading score
        e: embedding (or TF-IDF fallback) score
        """
        cfg = self.config
        return cfg.number_weight * n + cfg.heading_weight * h + cfg.embedding_weight * e

    # ------------------------------------------------------------------
    # Main matcher
    # ------------------------------------------------------------------

    def match(
        self,
        features_a: list[ClauseFeatures],
        features_b: list[ClauseFeatures],
    ) -> list[ClauseMatch]:
        """
        Align clauses from document A to document B using the hybrid score.

        Uses ``linear_sum_assignment`` to enforce one-to-one matching.
        Detects potential one-to-many restructuring by checking for clusters
        of near-threshold scores and flags them for review.

        Parameters
        ----------
        features_a, features_b:
            Feature lists for the two documents.

        Returns
        -------
        list[ClauseMatch]
            Full match list with per-component score breakdown, match type,
            and review flags.
        """
        if not features_a and not features_b:
            return []

        if not features_a:
            return [
                ClauseMatch(
                    clause_a=f.clause,
                    clause_b=None,
                    match_type=MatchType.ADDED_CANDIDATE,
                )
                for f in features_b
            ]

        if not features_b:
            return [
                ClauseMatch(
                    clause_a=f.clause,
                    clause_b=None,
                    match_type=MatchType.REMOVED_CANDIDATE,
                )
                for f in features_a
            ]

        n = len(features_a)
        m = len(features_b)

        # Pre-compute all per-component scores.
        num_scores = np.zeros((n, m))
        head_scores = np.zeros((n, m))
        emb_scores = np.zeros((n, m))
        combined = np.zeros((n, m))

        for i, fa in enumerate(features_a):
            for j, fb in enumerate(features_b):
                ns = self._number_score(fa.clause, fb.clause)
                hs = self._heading_score(fa.clause, fb.clause)
                es = self._embedding_score(fa, fb)
                cs = self._combined_score(ns, hs, es)
                num_scores[i, j] = ns
                head_scores[i, j] = hs
                emb_scores[i, j] = es
                combined[i, j] = cs

        # Hungarian algorithm for one-to-one assignment.
        row_ind, col_ind = linear_sum_assignment(-combined)

        matched_a: set[int] = set()
        matched_b: set[int] = set()
        results: list[ClauseMatch] = []

        for r, c in zip(row_ind, col_ind):
            score = float(combined[r, c])
            ns = float(num_scores[r, c])
            hs = float(head_scores[r, c])
            es = float(emb_scores[r, c])
            tfidf_s = cosine_sim(
                np.asarray(features_a[r].tfidf_vector),
                np.asarray(features_b[c].tfidf_vector),
            )

            if score >= self.config.accept_threshold:
                match_type = MatchType.HYBRID
                needs_review = False
                review_reason = ""
            elif score >= self.config.review_threshold:
                match_type = MatchType.NEEDS_REVIEW
                needs_review = True
                review_reason = (
                    f"Match score {score:.3f} is below accept threshold "
                    f"({self.config.accept_threshold}) but above review threshold "
                    f"({self.config.review_threshold}). Manual verification recommended."
                )
            else:
                # Score too low — do not create a match.
                continue

            # One-to-many restructuring detection: check whether clause_a
            # also scores near-threshold against other clauses in B.
            restructure_candidates = int(
                np.sum(combined[r, :] >= self.config.review_threshold)
            )
            if restructure_candidates > 1:
                needs_review = True
                review_reason += (
                    f" Possible 1-to-many restructuring: clause matches "
                    f"{restructure_candidates} candidates in document B."
                )

            results.append(
                ClauseMatch(
                    clause_a=features_a[r].clause,
                    clause_b=features_b[c].clause,
                    match_score=score,
                    number_score=ns,
                    heading_score=hs,
                    embedding_score=es,
                    tfidf_score=tfidf_s,
                    match_type=match_type,
                    needs_review=needs_review,
                    review_reason=review_reason.strip(),
                )
            )
            matched_a.add(r)
            matched_b.add(c)

        # Unmatched A → removed candidates.
        for i, fa in enumerate(features_a):
            if i not in matched_a:
                results.append(
                    ClauseMatch(
                        clause_a=fa.clause,
                        clause_b=None,
                        match_type=MatchType.REMOVED_CANDIDATE,
                    )
                )

        # Unmatched B → added candidates.
        for j, fb in enumerate(features_b):
            if j not in matched_b:
                results.append(
                    ClauseMatch(
                        clause_a=fb.clause,
                        clause_b=None,
                        match_type=MatchType.ADDED_CANDIDATE,
                    )
                )

        return results
