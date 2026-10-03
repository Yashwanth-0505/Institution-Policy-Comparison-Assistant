"""
config.py — PipelineConfig and load_config() for the POLICYX ML pipeline.
Reads from environment variables with fallback to defaults.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file if present (no-op when not found)
load_dotenv()


class PipelineConfig(BaseModel):
    """All tunable parameters for the pipeline with safe defaults."""

    # LLM providers
    gemini_model: str = Field(default="gemini-3.8-flash", description="Gemini model name")
    groq_model: str = Field(default="qwen/qwen3.8-27b", description="Groq model name")
    summary_timeout: int = Field(default=10, description="Per-provider timeout in seconds")

    # Hybrid matcher weights (must sum to 1.0)
    number_weight: float = Field(default=0.20, description="Weight for clause-number score")
    heading_weight: float = Field(default=0.30, description="Weight for heading similarity score")
    embedding_weight: float = Field(default=0.50, description="Weight for embedding cosine score")

    # Match thresholds
    accept_threshold: float = Field(default=0.50, description="Min score to accept a match")
    review_threshold: float = Field(
        default=0.35, description="Min score to flag for human review"
    )

    # Embedding model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model identifier",
    )

    # Segmentation
    min_clause_length: int = Field(
        default=20, description="Minimum chars for a clause to not be flagged short"
    )


def load_config() -> PipelineConfig:
    """
    Build a PipelineConfig from environment variables, falling back to defaults.

    Environment variable names mirror field names in upper-case, prefixed with
    ``POLICYX_``. Example: ``POLICYX_ACCEPT_THRESHOLD=0.60``.
    """
    env_overrides: dict[str, object] = {}

    _str_fields = ("gemini_model", "groq_model", "embedding_model")
    _int_fields = ("summary_timeout", "min_clause_length")
    _float_fields = (
        "number_weight",
        "heading_weight",
        "embedding_weight",
        "accept_threshold",
        "review_threshold",
    )

    for field in _str_fields:
        val = os.environ.get(f"POLICYX_{field.upper()}")
        if val is not None:
            env_overrides[field] = val

    for field in _int_fields:
        val = os.environ.get(f"POLICYX_{field.upper()}")
        if val is not None:
            env_overrides[field] = int(val)

    for field in _float_fields:
        val = os.environ.get(f"POLICYX_{field.upper()}")
        if val is not None:
            env_overrides[field] = float(val)

    return PipelineConfig(**env_overrides)
