"""
pipeline.py — Top-level orchestration for the POLICYX ML pipeline.

``compare_policies`` runs all 8 stages in order, collects per-stage errors
without crashing the whole pipeline, and returns a fully serialisable
:class:`~ml_pipeline.models.ComparisonResult`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ml_pipeline.aligner import BaselineMatcher, HybridMatcher
from ml_pipeline.citations import attach_citations
from ml_pipeline.config import PipelineConfig, load_config
from ml_pipeline.detector import detect_changes
from ml_pipeline.extractor import extract_document
from ml_pipeline.features import FeatureExtractor
from ml_pipeline.models import (
    ChangeResult,
    ClauseFeatures,
    ClauseMatch,
    ComparisonResult,
    DocumentExtraction,
    ExtractedClause,
    MatchType,
)
from ml_pipeline.segmenter import segment_clauses
from ml_pipeline.summarizer import generate_summaries

logger = logging.getLogger(__name__)


def compare_policies(
    policy_a_path: str | Path,
    policy_b_path: str | Path,
    config: Optional[PipelineConfig] = None,
) -> ComparisonResult:
    """
    Compare two policy PDF documents end-to-end.

    Runs the following 8-stage pipeline:

    1. **Extract** — parse both PDFs into :class:`~ml_pipeline.models.DocumentExtraction`
    2. **Segment** — split pages into :class:`~ml_pipeline.models.ExtractedClause` lists
    3. **Features** — fit TF-IDF and compute embeddings for both clause sets
    4. **Align** — run both :class:`~ml_pipeline.aligner.BaselineMatcher` and
       :class:`~ml_pipeline.aligner.HybridMatcher`
    5. **Detect** — classify changes from the hybrid matches
    6. **Summarise** — generate plain-English summaries via the LLM provider chain
    7. **Cite** — attach validated citations to every change result
    8. **Assemble** — build and return the :class:`~ml_pipeline.models.ComparisonResult`

    Each stage is wrapped in ``try/except``.  Non-fatal errors are appended to
    ``result.errors`` and execution continues where possible.  The final
    ``processing_complete`` flag is ``True`` only when stages 1–7 all succeeded.

    Parameters
    ----------
    policy_a_path:
        Path to the first (baseline) policy PDF.
    policy_b_path:
        Path to the second (comparison) policy PDF.
    config:
        Pipeline configuration.  If *None*, defaults are loaded from the
        environment.

    Returns
    -------
    ComparisonResult
        Fully populated result whose ``.model_dump_json()`` produces valid JSON.
    """
    if config is None:
        config = load_config()

    errors: list[str] = []
    warnings: list[str] = []

    # Sentinels — populated stage by stage.
    doc_a: Optional[DocumentExtraction] = None
    doc_b: Optional[DocumentExtraction] = None
    clauses_a: list[ExtractedClause] = []
    clauses_b: list[ExtractedClause] = []
    baseline_matches: list[ClauseMatch] = []
    hybrid_matches: list[ClauseMatch] = []
    changes: list[ChangeResult] = []
    stage_errors: list[bool] = [False] * 8  # True = stage failed

    # ------------------------------------------------------------------
    # Stage 1 — Extraction
    # ------------------------------------------------------------------
    logger.info("Stage 1: Extracting documents…")
    try:
        doc_a = extract_document(
            pdf_path=policy_a_path,
            document_id="doc_a",
            document_name=Path(policy_a_path).name,
        )
        warnings.extend(doc_a.warnings)
        errors.extend(doc_a.errors)
    except FileNotFoundError as exc:
        errors.append(f"Stage 1 — document A not found: {exc}")
        stage_errors[0] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 1 — failed to extract document A: {exc}")
        stage_errors[0] = True

    try:
        doc_b = extract_document(
            pdf_path=policy_b_path,
            document_id="doc_b",
            document_name=Path(policy_b_path).name,
        )
        warnings.extend(doc_b.warnings)
        errors.extend(doc_b.errors)
    except FileNotFoundError as exc:
        errors.append(f"Stage 1 — document B not found: {exc}")
        stage_errors[0] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 1 — failed to extract document B: {exc}")
        stage_errors[0] = True

    # If either document failed to extract we can still return a partial result.
    if doc_a is None:
        # Create a placeholder so the result is still serialisable.
        from ml_pipeline.models import DocumentExtraction as _DE
        doc_a = _DE(document_id="doc_a", document_name=str(policy_a_path))
    if doc_b is None:
        from ml_pipeline.models import DocumentExtraction as _DE
        doc_b = _DE(document_id="doc_b", document_name=str(policy_b_path))

    # ------------------------------------------------------------------
    # Stage 2 — Segmentation
    # ------------------------------------------------------------------
    logger.info("Stage 2: Segmenting clauses…")
    try:
        clauses_a = segment_clauses(doc_a, config=config)
        logger.info("  doc_a: %d clauses", len(clauses_a))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 2 — segmentation failed for document A: {exc}")
        stage_errors[1] = True

    try:
        clauses_b = segment_clauses(doc_b, config=config)
        logger.info("  doc_b: %d clauses", len(clauses_b))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 2 — segmentation failed for document B: {exc}")
        stage_errors[1] = True

    # ------------------------------------------------------------------
    # Stage 3 — Feature extraction
    # ------------------------------------------------------------------
    logger.info("Stage 3: Extracting features…")
    features_a = []
    features_b = []
    try:
        extractor = FeatureExtractor(config=config)
        combined = clauses_a + clauses_b
        if combined:
            # IMPORTANT: fit TF-IDF on the FULL combined corpus so both
            # document feature matrices share the same vocabulary and shape.
            extractor.fit_tfidf(combined)

            if clauses_a:
                tfidf_a = extractor.get_tfidf_matrix(clauses_a)
                embs_a  = extractor.get_embeddings(clauses_a)
                features_a = []
                for i, clause in enumerate(clauses_a):
                    features_a.append(
                        __import__('ml_pipeline.models', fromlist=['ClauseFeatures']).ClauseFeatures(
                            clause=clause,
                            tfidf_vector=tfidf_a[i],
                            embedding=embs_a[i] if embs_a is not None else None,
                        )
                    )

            if clauses_b:
                tfidf_b = extractor.get_tfidf_matrix(clauses_b)
                embs_b  = extractor.get_embeddings(clauses_b)
                features_b = []
                for i, clause in enumerate(clauses_b):
                    features_b.append(
                        __import__('ml_pipeline.models', fromlist=['ClauseFeatures']).ClauseFeatures(
                            clause=clause,
                            tfidf_vector=tfidf_b[i],
                            embedding=embs_b[i] if embs_b is not None else None,
                        )
                    )

        logger.info("  features_a: %d  features_b: %d", len(features_a), len(features_b))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 3 — feature extraction failed: {exc}")
        stage_errors[2] = True

    # ------------------------------------------------------------------
    # Stage 4 — Alignment (both matchers always run)
    # ------------------------------------------------------------------
    logger.info("Stage 4: Aligning clauses…")
    try:
        baseline_matcher = BaselineMatcher(config=config)
        baseline_matches = baseline_matcher.match(features_a, features_b)
        logger.info("  Baseline: %d matches", len(baseline_matches))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 4 — baseline matching failed: {exc}")
        stage_errors[3] = True

    try:
        hybrid_matcher = HybridMatcher(config=config)
        hybrid_matches = hybrid_matcher.match(features_a, features_b)
        logger.info("  Hybrid: %d matches", len(hybrid_matches))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 4 — hybrid matching failed: {exc}")
        stage_errors[3] = True

    # ------------------------------------------------------------------
    # Stage 5 — Change detection (on hybrid matches)
    # ------------------------------------------------------------------
    logger.info("Stage 5: Detecting changes…")
    try:
        changes = detect_changes(hybrid_matches)
        logger.info("  %d change results", len(changes))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 5 — change detection failed: {exc}")
        stage_errors[4] = True

    # ------------------------------------------------------------------
    # Stage 6 — Summary generation
    # ------------------------------------------------------------------
    logger.info("Stage 6: Generating summaries…")
    try:
        changes = generate_summaries(changes, config=config)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 6 — summary generation failed: {exc}")
        stage_errors[5] = True

    # ------------------------------------------------------------------
    # Stage 7 — Citations
    # ------------------------------------------------------------------
    logger.info("Stage 7: Attaching citations…")
    try:
        changes = attach_citations(changes, clauses_a, clauses_b)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stage 7 — citation attachment failed: {exc}")
        stage_errors[6] = True

    # ------------------------------------------------------------------
    # Stage 8 — Assemble result
    # ------------------------------------------------------------------
    logger.info("Stage 8: Assembling result…")

    added_clauses: list[ExtractedClause] = []
    removed_clauses: list[ExtractedClause] = []
    for match in hybrid_matches:
        if match.match_type == MatchType.ADDED_CANDIDATE:
            added_clauses.append(match.clause_a)
        elif match.match_type == MatchType.REMOVED_CANDIDATE:
            removed_clauses.append(match.clause_a)

    processing_complete = not any(stage_errors[:7])

    result = ComparisonResult(
        document_a=doc_a,
        document_b=doc_b,
        clauses_a=clauses_a,
        clauses_b=clauses_b,
        matches=hybrid_matches,
        added_clauses=added_clauses,
        removed_clauses=removed_clauses,
        changes=changes,
        warnings=warnings,
        errors=errors,
        processing_complete=processing_complete,
    )

    # Validate JSON serialisability before returning.
    try:
        result.model_dump_json()
    except Exception as exc:  # noqa: BLE001
        logger.error("Result is not JSON-serialisable: %s", exc)
        errors.append(f"Stage 8 — JSON serialisation failed: {exc}")
        result.errors = errors

    return result
