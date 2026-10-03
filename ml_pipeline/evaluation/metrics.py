"""
metrics.py — Evaluation metrics for the POLICYX ML pipeline.

All functions accept plain Python lists so they can be called independently
of the rest of the pipeline (useful in notebooks and unit tests).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def alignment_metrics(
    predicted: list[bool],
    gold: list[bool],
) -> dict[str, Any]:
    """
    Compute precision, recall, and F1 for clause alignment (binary matching).

    A *True* value means "this pair should be matched"; *False* means it
    should not.

    Parameters
    ----------
    predicted:
        Model predictions — one ``bool`` per pair.
    gold:
        Ground-truth labels — same length as *predicted*.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``,
        ``correct_count``, ``total_count``.

    Raises
    ------
    ValueError
        If *predicted* and *gold* have different lengths.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, gold={len(gold)}"
        )

    tp = sum(1 for p, g in zip(predicted, gold) if p and g)
    fp = sum(1 for p, g in zip(predicted, gold) if p and not g)
    fn = sum(1 for p, g in zip(predicted, gold) if not p and g)
    correct = sum(1 for p, g in zip(predicted, gold) if p == g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct_count": correct,
        "total_count": len(gold),
    }


def classification_metrics(
    predicted: list[str],
    gold: list[str],
) -> dict[str, Any]:
    """
    Compute per-class and macro-averaged F1 for change-type classification.

    Parameters
    ----------
    predicted:
        Predicted change-type labels (e.g. ``"unchanged"``, ``"substantive"``).
    gold:
        Ground-truth labels — same length as *predicted*.

    Returns
    -------
    dict
        Keys: ``per_class`` (dict of label → {precision, recall, f1, support}),
        ``macro_f1``, ``confusion_matrix`` (dict of gold → predicted → count),
        ``labels`` (sorted list of all classes seen).
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, gold={len(gold)}"
        )

    labels = sorted(set(gold) | set(predicted))

    # Confusion matrix: gold_label -> predicted_label -> count
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1

    per_class: dict[str, dict[str, Any]] = {}
    f1_scores: list[float] = []

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        support = sum(confusion[label].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Convert defaultdicts to plain dicts for clean serialisation.
    confusion_plain = {g: dict(row) for g, row in confusion.items()}

    return {
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": confusion_plain,
        "labels": labels,
    }


def diagnostics(
    results: list[dict],
    gold_matches: list[bool],
) -> dict[str, Any]:
    """
    Compute diagnostic statistics over a set of change results.

    Parameters
    ----------
    results:
        List of dicts, each representing a
        :class:`~ml_pipeline.models.ChangeResult` serialised as a dict.
        Expected keys: ``change_type``, ``needs_review``, ``citations``,
        ``match`` (dict with ``match_type``).
    gold_matches:
        Ground-truth match flags (same order as *results*).

    Returns
    -------
    dict
        Keys: ``false_matches``, ``missed_matches``, ``unmatched_count``,
        ``ambiguous_count``, ``needs_review_count``,
        ``citation_integrity_rate``.
    """
    false_matches = 0
    missed_matches = 0
    unmatched_count = 0
    ambiguous_count = 0
    needs_review_count = 0
    total_citations = 0
    valid_citations = 0

    for result, gold in zip(results, gold_matches):
        change_type = result.get("change_type", "")
        match = result.get("match", {})
        match_type = match.get("match_type", "")
        predicted_matched = match_type not in (
            "added_candidate", "removed_candidate", "unmatched"
        )

        if predicted_matched and not gold:
            false_matches += 1
        if not predicted_matched and gold:
            missed_matches += 1
        if match_type in ("added_candidate", "removed_candidate", "unmatched"):
            unmatched_count += 1
        if change_type == "uncertain":
            ambiguous_count += 1
        if result.get("needs_review", False):
            needs_review_count += 1

        for citation in result.get("citations", []):
            total_citations += 1
            if citation.get("citation_valid", False):
                valid_citations += 1

    citation_integrity = (
        valid_citations / total_citations if total_citations > 0 else 1.0
    )

    return {
        "false_matches": false_matches,
        "missed_matches": missed_matches,
        "unmatched_count": unmatched_count,
        "ambiguous_count": ambiguous_count,
        "needs_review_count": needs_review_count,
        "citation_integrity_rate": round(citation_integrity, 4),
    }
