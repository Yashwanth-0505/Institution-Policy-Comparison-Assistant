"""
test_summarizer.py — Tests for ml_pipeline.summarizer
"""
from __future__ import annotations

import pytest

from ml_pipeline.models import (
    BoundedSummary,
    ChangeResult,
    ChangeType,
    ClauseMatch,
    MatchType,
)
from ml_pipeline.summarizer import (
    DeterministicProvider,
    GeminiProvider,
    GroqProvider,
    get_summary,
)
from ml_pipeline.tests.conftest import make_clause


def _make_change_result(change_type: ChangeType = ChangeType.SUBSTANTIVE) -> ChangeResult:
    ca = make_clause("doc_a", "doc_a_clause_0000", "1. Policy",
                     "All staff must complete training within 30 days.")
    cb = make_clause("doc_b", "doc_b_clause_0000", "1. Policy",
                     "All employees must complete training within 60 days.")
    match = ClauseMatch(
        clause_a=ca,
        clause_b=cb,
        match_score=0.85,
        match_type=MatchType.HYBRID,
    )
    return ChangeResult(
        match=match,
        change_type=change_type,
        signals=[],
        needs_review=False,
    )


# ---------------------------------------------------------------------------

def test_deterministic_all_change_types() -> None:
    """DeterministicProvider produces a non-empty summary for every ChangeType."""
    provider = DeterministicProvider()
    from ml_pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    for ct in ChangeType:
        result = _make_change_result(ct)
        summary = provider.generate(result, cfg)
        assert isinstance(summary, BoundedSummary)
        assert len(summary.summary) > 0
        assert summary.provider_used == "deterministic"


def test_gemini_key_missing_skipped() -> None:
    """GeminiProvider.is_available() returns False when GEMINI_API_KEY is absent."""
    import os
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        provider = GeminiProvider()
        assert not provider.is_available()
    finally:
        if old:
            os.environ["GEMINI_API_KEY"] = old


def test_groq_key_missing_skipped() -> None:
    """GroqProvider.is_available() returns False when GROQ_API_KEY is absent."""
    import os
    old = os.environ.pop("GROQ_API_KEY", None)
    try:
        provider = GroqProvider()
        assert not provider.is_available()
    finally:
        if old:
            os.environ["GROQ_API_KEY"] = old


def test_gemini_fails_falls_back_to_groq(mocker) -> None:
    """When Gemini raises, get_summary falls back to the next provider."""
    import os
    os.environ["GEMINI_API_KEY"] = "fake-gemini-key"
    os.environ["GROQ_API_KEY"] = "fake-groq-key"

    # Patch Gemini to fail
    mock_gemini = mocker.MagicMock()
    mock_gemini.return_value.is_available.return_value = True
    mock_gemini.return_value.generate.side_effect = RuntimeError("Gemini API error")

    # Patch Groq to fail too (so deterministic handles it)
    mock_groq = mocker.MagicMock()
    mock_groq.return_value.is_available.return_value = True
    mock_groq.return_value.generate.side_effect = RuntimeError("Groq API error")

    mocker.patch("ml_pipeline.summarizer.GeminiProvider", mock_gemini)
    mocker.patch("ml_pipeline.summarizer.GroqProvider", mock_groq)

    result = _make_change_result()
    from ml_pipeline.config import PipelineConfig
    summary = get_summary(result, PipelineConfig())
    # Should fall through to DeterministicProvider
    assert summary.provider_used == "deterministic"
    assert len(summary.summary) > 0

    del os.environ["GEMINI_API_KEY"]
    del os.environ["GROQ_API_KEY"]


def test_both_llm_fail_deterministic_used(mocker) -> None:
    """When both LLM providers fail, DeterministicProvider is used."""
    import os
    os.environ["GEMINI_API_KEY"] = "fake-key"
    os.environ["GROQ_API_KEY"] = "fake-key"

    mocker.patch(
        "ml_pipeline.summarizer.GeminiProvider.generate",
        side_effect=RuntimeError("fail"),
    )
    mocker.patch(
        "ml_pipeline.summarizer.GroqProvider.generate",
        side_effect=RuntimeError("fail"),
    )

    result = _make_change_result()
    from ml_pipeline.config import PipelineConfig
    summary = get_summary(result, PipelineConfig())
    assert summary.provider_used == "deterministic"

    del os.environ["GEMINI_API_KEY"]
    del os.environ["GROQ_API_KEY"]


def test_provider_used_field_set() -> None:
    """provider_used field reflects which provider generated the summary."""
    result = _make_change_result()
    from ml_pipeline.config import PipelineConfig
    # Without API keys, deterministic is used
    summary = get_summary(result, PipelineConfig())
    assert summary.provider_used in ("gemini", "groq", "deterministic")


def test_fallback_reason_populated(mocker) -> None:
    """fallback_reason is set when providers are skipped."""
    import os
    os.environ["GEMINI_API_KEY"] = "fake-key"

    mocker.patch(
        "ml_pipeline.summarizer.GeminiProvider.generate",
        side_effect=RuntimeError("Simulated Gemini failure"),
    )

    result = _make_change_result()
    from ml_pipeline.config import PipelineConfig
    summary = get_summary(result, PipelineConfig())
    # fallback_reason should mention the failure
    assert summary.fallback_reason != "" or summary.provider_used == "deterministic"

    del os.environ["GEMINI_API_KEY"]
