"""
ml_pipeline — POLICYX Institution Policy Comparison ML Pipeline.
Public API exports.
"""
from ml_pipeline.aligner import BaselineMatcher, HybridMatcher
from ml_pipeline.citations import attach_citations, validate_citation
from ml_pipeline.config import PipelineConfig, load_config
from ml_pipeline.detector import detect_changes
from ml_pipeline.extractor import extract_document
from ml_pipeline.features import FeatureExtractor
from ml_pipeline.pipeline import compare_policies
from ml_pipeline.segmenter import segment_clauses
from ml_pipeline.summarizer import generate_summaries, get_summary

__all__ = [
    "compare_policies",
    "extract_document",
    "segment_clauses",
    "FeatureExtractor",
    "BaselineMatcher",
    "HybridMatcher",
    "detect_changes",
    "generate_summaries",
    "get_summary",
    "attach_citations",
    "validate_citation",
    "PipelineConfig",
    "load_config",
]
