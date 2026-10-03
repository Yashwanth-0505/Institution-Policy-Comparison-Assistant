"""
test_evaluation.py — Tests for the evaluation suite.
"""
from __future__ import annotations

import pytest

from ml_pipeline.evaluation.dataset import load_eval_dataset, load_tuning_dataset
from ml_pipeline.evaluation.metrics import alignment_metrics, classification_metrics
from ml_pipeline.evaluation.runner import run_evaluation


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_load_eval_dataset_30_minimum() -> None:
    """Evaluation dataset has at least 30 entries."""
    data = load_eval_dataset()
    assert len(data) >= 30


def test_load_tuning_dataset_15_minimum() -> None:
    """Tuning dataset has at least 15 entries."""
    data = load_tuning_dataset()
    assert len(data) >= 15


def test_no_id_overlap() -> None:
    """Eval and tuning datasets share no IDs."""
    eval_ids = {d["id"] for d in load_eval_dataset()}
    tune_ids = {d["id"] for d in load_tuning_dataset()}
    overlap = eval_ids & tune_ids
    assert len(overlap) == 0, f"Overlapping IDs found: {overlap}"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_alignment_metrics_perfect() -> None:
    """Perfect predictions produce precision=recall=f1=1.0."""
    gold = [True, True, False, True, False]
    pred = [True, True, False, True, False]
    m = alignment_metrics(pred, gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["correct_count"] == 5


def test_alignment_metrics_all_wrong() -> None:
    """All-wrong predictions produce f1=0.0."""
    gold = [True, True, True]
    pred = [False, False, False]
    m = alignment_metrics(pred, gold)
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_classification_smoke() -> None:
    """classification_metrics runs without error and returns expected keys."""
    gold = ["unchanged", "substantive", "wording_only", "substantive", "unchanged"]
    pred = ["unchanged", "wording_only", "wording_only", "substantive", "unchanged"]
    m = classification_metrics(pred, gold)
    assert "per_class" in m
    assert "macro_f1" in m
    assert "confusion_matrix" in m
    assert "labels" in m
    assert isinstance(m["macro_f1"], float)


def test_runner_returns_report() -> None:
    """run_evaluation() returns an EvaluationReport with expected fields."""
    from ml_pipeline.evaluation.runner import EvaluationReport
    report = run_evaluation()
    assert isinstance(report, EvaluationReport)
    assert report.labelled_pairs_count >= 30
    assert 0.0 <= report.matching_accuracy <= 1.0
    assert report.wording_only_test is not None
    assert report.baseline is not None
    assert report.hybrid is not None


# ---------------------------------------------------------------------------
# Day 2 sprint tests
# ---------------------------------------------------------------------------

def test_wording_only_pairs_exist_in_dataset() -> None:
    """Dataset contains at least 3 wording_only pairs (Day 2 sprint requirement)."""
    data = load_eval_dataset()
    wording_only = [d for d in data if d.get("expected_change_type") == "wording_only"]
    assert len(wording_only) >= 3, (
        f"Expected at least 3 wording_only pairs, found {len(wording_only)}"
    )


def test_wording_only_correctly_classified() -> None:
    """
    Day 2 sprint: wording_only pairs must NOT be classified as substantive.

    Runs the evaluation and asserts that no wording-only clause pair is
    incorrectly labelled as substantive.
    """
    report = run_evaluation()
    wo = report.wording_only_test
    assert wo.total_wording_only_pairs >= 3, (
        f"Expected at least 3 wording_only pairs in dataset, found {wo.total_wording_only_pairs}"
    )
    assert wo.incorrectly_classified_as_substantive == 0, (
        f"Day 2 sprint FAIL: {wo.incorrectly_classified_as_substantive} wording-only pair(s) "
        "were incorrectly classified as substantive."
    )
    assert wo.passed, "Day 2 sprint wording-only detection test did not pass."
