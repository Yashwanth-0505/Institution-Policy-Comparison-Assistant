"""
dataset.py — Dataset loaders for the POLICYX evaluation suite.
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


def load_eval_dataset() -> list[dict]:
    """
    Load the hand-labelled evaluation dataset.

    Returns
    -------
    list[dict]
        30+ labelled clause pairs.  Each entry has keys:
        ``id``, ``category``, ``clause_a_text``, ``clause_b_text``,
        ``expected_match``, ``expected_change_type``, ``notes``.

    Raises
    ------
    FileNotFoundError
        If ``evaluation/data/labelled_pairs.json`` does not exist.
    """
    path = _DATA_DIR / "labelled_pairs.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_tuning_dataset() -> list[dict]:
    """
    Load the weight-tuning dataset (kept strictly separate from evaluation).

    Returns
    -------
    list[dict]
        15 clause pairs for hyperparameter tuning.  Same schema as the eval
        dataset but with IDs starting with ``tune_``.

    Raises
    ------
    FileNotFoundError
        If ``evaluation/data/tuning_pairs.json`` does not exist.
    """
    path = _DATA_DIR / "tuning_pairs.json"
    if not path.exists():
        raise FileNotFoundError(f"Tuning dataset not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
