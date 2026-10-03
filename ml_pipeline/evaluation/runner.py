"""
runner.py — Evaluation runner for the POLICYX ML pipeline.

Loads the hand-labelled dataset, runs both matchers on each pair,
computes metrics, and returns a structured :class:`EvaluationReport`.

Day 2 sprint: reports wording-only detection accuracy explicitly.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from ml_pipeline.aligner import BaselineMatcher, HybridMatcher
from ml_pipeline.config import PipelineConfig, load_config
from ml_pipeline.detector import detect_changes
from ml_pipeline.evaluation.dataset import load_eval_dataset
from ml_pipeline.evaluation.metrics import alignment_metrics, classification_metrics
from ml_pipeline.features import FeatureExtractor
from ml_pipeline.models import ExtractedClause, MatchType, SectionMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

class MatcherReport(BaseModel):
    """Per-matcher evaluation metrics."""
    name: str
    precision: float
    recall: float
    f1: float
    correct_count: int
    total_count: int
    macro_classification_f1: float
    per_class_f1: dict[str, float] = Field(default_factory=dict)


class WordingOnlyTest(BaseModel):
    """Day 2 sprint: wording-only detection test result."""
    total_wording_only_pairs: int
    correctly_classified_as_wording_only: int
    incorrectly_classified_as_substantive: int
    passed: bool


class EvaluationReport(BaseModel):
    """Complete evaluation report for the POLICYX ML pipeline."""
    dataset_size: int
    labelled_pairs_count: int
    correctly_matched_count: int
    matching_accuracy: float
    wording_only_test: WordingOnlyTest
    baseline: MatcherReport
    hybrid: MatcherReport
    limitations: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval_clause(
    text: str,
    doc_id: str,
    clause_idx: int,
    heading: str = "",
) -> ExtractedClause:
    """Create a minimal ExtractedClause from raw text for evaluation purposes."""
    return ExtractedClause(
        document_id=doc_id,
        document_name=doc_id,
        version="eval",
        clause_id=f"{doc_id}_clause_{clause_idx:04d}",
        heading=heading,
        original_text=text,
        normalized_text=text,
        page_number=1,
        section_metadata=SectionMetadata(),
        flagged_for_review=False,
        review_reason="",
    )


def _run_matcher_on_dataset(
    pairs: list[dict],
    matcher_name: str,
    config: PipelineConfig,
) -> tuple[list[bool], list[bool], list[str], list[str]]:
    """
    Run a named matcher on every pair in the dataset.

    Returns
    -------
    tuple of (predicted_match, gold_match, predicted_type, gold_type)
    """
    extractor = FeatureExtractor(config=config)

    predicted_match: list[bool] = []
    gold_match: list[bool] = []
    predicted_type: list[str] = []
    gold_type: list[str] = []

    for idx, pair in enumerate(pairs):
        text_a = pair.get("clause_a_text", "")
        text_b = pair.get("clause_b_text", "")
        gold_matched: bool = pair.get("expected_match", False)
        gold_change: str = pair.get("expected_change_type", "uncertain")

        gold_match.append(gold_matched)
        gold_type.append(gold_change)

        # Skip pairs with empty sides (added/removed).
        if not text_a or not text_b:
            predicted_match.append(False)
            if not text_a:
                predicted_type.append("added")
            else:
                predicted_type.append("removed")
            continue

        clause_a = _make_eval_clause(text_a, f"eval_a_{idx}", idx)
        clause_b = _make_eval_clause(text_b, f"eval_b_{idx}", idx)

        # Each pair is featurised independently to avoid cross-pair TF-IDF leakage.
        feats = extractor.extract([clause_a, clause_b])
        fa, fb = feats[0], feats[1]

        if matcher_name == "baseline":
            matcher: Any = BaselineMatcher(config=config)
        else:
            matcher = HybridMatcher(config=config)

        matches = matcher.match([fa], [fb])
        changes = detect_changes(matches)

        if not matches or not changes:
            predicted_match.append(False)
            predicted_type.append("uncertain")
            continue

        primary = matches[0]
        is_matched = primary.match_type not in (
            MatchType.ADDED_CANDIDATE,
            MatchType.REMOVED_CANDIDATE,
            MatchType.UNMATCHED,
        )
        predicted_match.append(is_matched)
        predicted_type.append(changes[0].change_type.value)

    return predicted_match, gold_match, predicted_type, gold_type


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(config: Optional[PipelineConfig] = None) -> EvaluationReport:
    """
    Run the full evaluation suite and return an :class:`EvaluationReport`.

    Loads ``evaluation/data/labelled_pairs.json``, runs both matchers, computes
    alignment and classification metrics, and reports the Day 2 wording-only
    detection test.

    Parameters
    ----------
    config:
        Pipeline configuration.  If *None*, defaults are used.

    Returns
    -------
    EvaluationReport
        Structured report with accuracy, per-class metrics, and Day 2 sprint
        compliance status.
    """
    if config is None:
        config = load_config()

    pairs = load_eval_dataset()
    n = len(pairs)

    # -----------------------------------------------------------------------
    # Run both matchers
    # -----------------------------------------------------------------------
    baseline_pred_match, gold_match, baseline_pred_type, gold_type = (
        _run_matcher_on_dataset(pairs, "baseline", config)
    )
    hybrid_pred_match, _, hybrid_pred_type, _ = (
        _run_matcher_on_dataset(pairs, "hybrid", config)
    )

    # -----------------------------------------------------------------------
    # Alignment metrics
    # -----------------------------------------------------------------------
    baseline_align = alignment_metrics(baseline_pred_match, gold_match)
    hybrid_align = alignment_metrics(hybrid_pred_match, gold_match)

    # -----------------------------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------------------------
    baseline_class = classification_metrics(baseline_pred_type, gold_type)
    hybrid_class = classification_metrics(hybrid_pred_type, gold_type)

    # -----------------------------------------------------------------------
    # Day 2 sprint: wording-only detection
    # -----------------------------------------------------------------------
    wording_pairs = [p for p in pairs if p.get("expected_change_type") == "wording_only"]
    total_wo = len(wording_pairs)
    correct_wo = 0
    wrong_wo = 0

    for i, pair in enumerate(pairs):
        if pair.get("expected_change_type") != "wording_only":
            continue
        pred = hybrid_pred_type[i]
        if pred == "wording_only":
            correct_wo += 1
        elif pred == "substantive":
            wrong_wo += 1

    wording_test = WordingOnlyTest(
        total_wording_only_pairs=total_wo,
        correctly_classified_as_wording_only=correct_wo,
        incorrectly_classified_as_substantive=wrong_wo,
        passed=(wrong_wo == 0 and total_wo > 0),
    )

    # -----------------------------------------------------------------------
    # Assemble matcher reports
    # -----------------------------------------------------------------------
    baseline_report = MatcherReport(
        name="baseline",
        precision=baseline_align["precision"],
        recall=baseline_align["recall"],
        f1=baseline_align["f1"],
        correct_count=baseline_align["correct_count"],
        total_count=baseline_align["total_count"],
        macro_classification_f1=baseline_class["macro_f1"],
        per_class_f1={
            label: stats["f1"]
            for label, stats in baseline_class["per_class"].items()
        },
    )

    hybrid_report = MatcherReport(
        name="hybrid",
        precision=hybrid_align["precision"],
        recall=hybrid_align["recall"],
        f1=hybrid_align["f1"],
        correct_count=hybrid_align["correct_count"],
        total_count=hybrid_align["total_count"],
        macro_classification_f1=hybrid_class["macro_f1"],
        per_class_f1={
            label: stats["f1"]
            for label, stats in hybrid_class["per_class"].items()
        },
    )

    correctly_matched = hybrid_align["correct_count"]
    matching_accuracy = correctly_matched / n if n > 0 else 0.0

    limitations = [
        "Evaluation uses synthetic clause pairs, not real PDF documents.",
        "Embedding model may not be cached offline; embeddings fall back to TF-IDF.",
        "LLM summaries are not evaluated (deterministic provider used in eval).",
        "Wording-only detection relies on difflib ratio; domain-specific paraphrases may score below threshold.",
        "Dataset size (30 pairs) is small; results may not generalise to all policy types.",
    ]

    report = EvaluationReport(
        dataset_size=n,
        labelled_pairs_count=n,
        correctly_matched_count=correctly_matched,
        matching_accuracy=round(matching_accuracy, 4),
        wording_only_test=wording_test,
        baseline=baseline_report,
        hybrid=hybrid_report,
        limitations=limitations,
    )

    # -----------------------------------------------------------------------
    # Print formatted report
    # -----------------------------------------------------------------------
    _print_report(report)

    return report


def _print_report(report: EvaluationReport) -> None:
    """Print a human-readable summary of the evaluation report."""
    wo = report.wording_only_test
    print("\n" + "=" * 60)
    print("POLICYX ML Pipeline — Evaluation Report")
    print("=" * 60)
    print(f"Timestamp : {report.timestamp}")
    print(f"Dataset   : {report.labelled_pairs_count} labelled pairs")
    print(
        f"Correctly matched (hybrid): {report.correctly_matched_count} / "
        f"{report.labelled_pairs_count} "
        f"(accuracy: {report.matching_accuracy * 100:.1f}%)"
    )
    print()
    print("── Wording-Only Detection (Day 2 Sprint) ──────────────")
    print(f"  Total wording-only pairs in dataset : {wo.total_wording_only_pairs}")
    print(f"  Correctly labelled as wording_only  : {wo.correctly_classified_as_wording_only}")
    print(f"  Incorrectly labelled as substantive : {wo.incorrectly_classified_as_substantive}")
    print(f"  Result : {'✅ PASS' if wo.passed else '❌ FAIL'}")
    print()
    print("── Matcher Comparison ──────────────────────────────────")
    print(f"  {'Metric':<28} {'Baseline':>10} {'Hybrid':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Precision':<28} {report.baseline.precision:>10.4f} {report.hybrid.precision:>10.4f}")
    print(f"  {'Recall':<28} {report.baseline.recall:>10.4f} {report.hybrid.recall:>10.4f}")
    print(f"  {'F1 (alignment)':<28} {report.baseline.f1:>10.4f} {report.hybrid.f1:>10.4f}")
    print(f"  {'Macro F1 (classification)':<28} {report.baseline.macro_classification_f1:>10.4f} {report.hybrid.macro_classification_f1:>10.4f}")
    print()
    print("── Known Limitations ───────────────────────────────────")
    for lim in report.limitations:
        print(f"  • {lim}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    report = run_evaluation()
    print(report.model_dump_json(indent=2))
