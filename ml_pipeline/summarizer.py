"""
summarizer.py — LLM-backed summary generation for the POLICYX ML pipeline.

Provider chain (tried in order):
  1. GeminiProvider  — uses google-genai SDK
  2. GroqProvider    — uses openai SDK pointed at Groq endpoint
  3. DeterministicProvider — always available, no API required

API keys are read exclusively from environment variables — never hardcoded.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
from abc import ABC, abstractmethod
from typing import Optional

from ml_pipeline.config import PipelineConfig, load_config
from ml_pipeline.models import BoundedSummary, ChangeResult, ChangeType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic templates
# ---------------------------------------------------------------------------

_DETERMINISTIC_TEMPLATES: dict[ChangeType, str] = {
    ChangeType.UNCHANGED: "This clause is identical in both document versions. No changes detected.",
    ChangeType.WORDING_ONLY: (
        "This clause has been reworded between versions but retains the same meaning. "
        "No substantive policy changes were detected."
    ),
    ChangeType.SUBSTANTIVE: (
        "This clause contains substantive changes that may affect policy obligations or rights."
    ),
    ChangeType.ADDED: "This clause was added in the new document version and was not present previously.",
    ChangeType.REMOVED: "This clause was present in the original document but has been removed.",
    ChangeType.UNCERTAIN: (
        "The relationship between these clauses is ambiguous. Manual review is recommended."
    ),
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SummaryProvider(ABC):
    """Abstract base class for summary providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return *True* when this provider can be called."""
        ...

    @abstractmethod
    def generate(
        self,
        change_result: ChangeResult,
        config: PipelineConfig,
    ) -> BoundedSummary:
        """
        Generate a :class:`~ml_pipeline.models.BoundedSummary` for *change_result*.

        Must raise an exception on any failure so the caller can fall back.
        """
        ...


# ---------------------------------------------------------------------------
# Deterministic provider (always works)
# ---------------------------------------------------------------------------

class DeterministicProvider(SummaryProvider):
    """Builds a summary from change_type and signals — no API required."""

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        change_result: ChangeResult,
        config: PipelineConfig,
    ) -> BoundedSummary:
        base = _DETERMINISTIC_TEMPLATES.get(
            change_result.change_type,
            "Change analysis result — see signals for details.",
        )

        signal_descriptions: list[str] = []
        for sig in change_result.signals:
            parts = [sig.description]
            if sig.old_value and sig.old_value != "(none)":
                parts.append(f"was: {sig.old_value}")
            if sig.new_value and sig.new_value != "(none)":
                parts.append(f"now: {sig.new_value}")
            signal_descriptions.append(" — ".join(parts))

        if signal_descriptions:
            signals_text = " Detected: " + "; ".join(signal_descriptions) + "."
        else:
            signals_text = ""

        summary = base + signals_text

        return BoundedSummary(
            summary=summary,
            classification_suggestion=change_result.change_type.value,
            confidence=0.85,
            provider_used="deterministic",
            fallback_reason="",
        )


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiProvider(SummaryProvider):
    """Uses google-genai SDK to generate summaries via Gemini."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _build_prompt(self, change_result: ChangeResult) -> str:
        match = change_result.match
        clause_a = match.clause_a
        clause_b = match.clause_b

        doc_name_a = clause_a.document_name
        clause_id_a = clause_a.clause_id
        page_a = clause_a.page_number
        old_text = clause_a.normalized_text

        if clause_b is not None:
            doc_name_b = clause_b.document_name
            clause_id_b = clause_b.clause_id
            page_b = clause_b.page_number
            new_text = clause_b.normalized_text
        else:
            doc_name_b = doc_name_a
            clause_id_b = clause_id_a
            page_b = page_a
            new_text = "(clause not present in new version)"

        signals_text = "; ".join(
            f"{s.signal_type}: {s.description}" for s in change_result.signals
        ) or "none"

        return (
            f'OLD CLAUSE ({doc_name_a}, Clause {clause_id_a}, Page {page_a}):\n'
            f'{old_text}\n\n'
            f'NEW CLAUSE ({doc_name_b}, Clause {clause_id_b}, Page {page_b}):\n'
            f'{new_text}\n\n'
            f'DETECTED SIGNALS: {signals_text}\n\n'
            'Respond ONLY in JSON:\n'
            '{"summary": "...", "classification_suggestion": '
            '"wording_only|substantive|uncertain", "confidence": "high|medium|low"}\n\n'
            'Do NOT invent citations, legal consequences, or information not in the text above.'
        )

    def generate(
        self,
        change_result: ChangeResult,
        config: PipelineConfig,
    ) -> BoundedSummary:
        import google.genai as genai  # type: ignore[import]

        client = genai.Client(api_key=self.api_key)
        prompt = self._build_prompt(change_result)

        result_holder: list[Optional[BoundedSummary]] = [None]
        error_holder: list[Optional[Exception]] = [None]

        def _call() -> None:
            try:
                response = client.models.generate_content(
                    model=config.gemini_model,
                    contents=prompt,
                )
                raw = response.text.strip()
                # Strip markdown code fences if present.
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                data = json.loads(raw)
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                result_holder[0] = BoundedSummary(
                    summary=str(data.get("summary", "")),
                    classification_suggestion=str(
                        data.get("classification_suggestion", "")
                    ),
                    confidence=confidence_map.get(
                        str(data.get("confidence", "medium")).lower(), 0.7
                    ),
                    provider_used="gemini",
                    fallback_reason="",
                )
            except Exception as exc:  # noqa: BLE001
                error_holder[0] = exc

        import re as _re  # local import to avoid shadowing module-level re

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=config.summary_timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Gemini API call timed out after {config.summary_timeout}s"
            )
        if error_holder[0] is not None:
            raise error_holder[0]
        if result_holder[0] is None:
            raise RuntimeError("Gemini returned no result")
        return result_holder[0]


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class GroqProvider(SummaryProvider):
    """Uses OpenAI-compatible SDK pointed at the Groq endpoint."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.environ.get("GROQ_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _build_messages(self, change_result: ChangeResult) -> list[dict]:
        match = change_result.match
        clause_a = match.clause_a
        clause_b = match.clause_b

        doc_name_a = clause_a.document_name
        clause_id_a = clause_a.clause_id
        page_a = clause_a.page_number
        old_text = clause_a.normalized_text

        if clause_b is not None:
            doc_name_b = clause_b.document_name
            clause_id_b = clause_b.clause_id
            page_b = clause_b.page_number
            new_text = clause_b.normalized_text
        else:
            doc_name_b = doc_name_a
            clause_id_b = clause_id_a
            page_b = page_a
            new_text = "(clause not present in new version)"

        signals_text = "; ".join(
            f"{s.signal_type}: {s.description}" for s in change_result.signals
        ) or "none"

        user_content = (
            f'OLD CLAUSE ({doc_name_a}, Clause {clause_id_a}, Page {page_a}):\n'
            f'{old_text}\n\n'
            f'NEW CLAUSE ({doc_name_b}, Clause {clause_id_b}, Page {page_b}):\n'
            f'{new_text}\n\n'
            f'DETECTED SIGNALS: {signals_text}\n\n'
            'Respond ONLY in JSON:\n'
            '{"summary": "...", "classification_suggestion": '
            '"wording_only|substantive|uncertain", "confidence": "high|medium|low"}\n\n'
            'Do NOT invent citations, legal consequences, or information not in the text above.'
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are a policy analysis assistant. Compare the two clause versions "
                    "and return a JSON summary. Be factual and concise."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        change_result: ChangeResult,
        config: PipelineConfig,
    ) -> BoundedSummary:
        import openai  # type: ignore[import]

        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
        )
        messages = self._build_messages(change_result)

        result_holder: list[Optional[BoundedSummary]] = [None]
        error_holder: list[Optional[Exception]] = [None]

        def _call() -> None:
            try:
                response = client.chat.completions.create(
                    model=config.groq_model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.1,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()
                import re as _re
                if raw.startswith("```"):
                    raw = _re.sub(r"^```[a-z]*\n?", "", raw)
                    raw = _re.sub(r"\n?```$", "", raw)
                data = json.loads(raw)
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                result_holder[0] = BoundedSummary(
                    summary=str(data.get("summary", "")),
                    classification_suggestion=str(
                        data.get("classification_suggestion", "")
                    ),
                    confidence=confidence_map.get(
                        str(data.get("confidence", "medium")).lower(), 0.7
                    ),
                    provider_used="groq",
                    fallback_reason="",
                )
            except Exception as exc:  # noqa: BLE001
                error_holder[0] = exc

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=config.summary_timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Groq API call timed out after {config.summary_timeout}s"
            )
        if error_holder[0] is not None:
            raise error_holder[0]
        if result_holder[0] is None:
            raise RuntimeError("Groq returned no result")
        return result_holder[0]


# ---------------------------------------------------------------------------
# Provider chain
# ---------------------------------------------------------------------------

PROVIDER_CHAIN: list[type[SummaryProvider]] = [
    GeminiProvider,
    GroqProvider,
    DeterministicProvider,
]


def get_summary(
    change_result: ChangeResult,
    config: Optional[PipelineConfig] = None,
) -> BoundedSummary:
    """
    Generate a summary for *change_result* using the provider chain.

    Tries each provider in :data:`PROVIDER_CHAIN` order.  Skips unavailable
    providers (missing API key).  Falls back to the next on any exception.
    The :class:`~ml_pipeline.models.DeterministicProvider` is always last and
    always succeeds.

    Parameters
    ----------
    change_result:
        The change analysis result to summarise.
    config:
        Pipeline configuration.  If *None*, defaults are loaded.

    Returns
    -------
    BoundedSummary
        Always returns a result; never raises.
    """
    if config is None:
        config = load_config()

    last_reason = ""
    for provider_cls in PROVIDER_CHAIN:
        provider = provider_cls()
        if not provider.is_available():
            last_reason = f"{provider_cls.__name__} not available (no API key)"
            logger.debug(last_reason)
            continue
        try:
            summary = provider.generate(change_result, config)
            if last_reason:
                summary.fallback_reason = last_reason
            return summary
        except Exception as exc:  # noqa: BLE001
            last_reason = f"{provider_cls.__name__} failed: {exc}"
            logger.warning("Provider fallback — %s", last_reason)

    # Should never reach here (DeterministicProvider always works).
    return BoundedSummary(
        summary="Summary unavailable.",
        provider_used="none",
        fallback_reason=last_reason,
    )


def generate_summaries(
    change_results: list[ChangeResult],
    config: Optional[PipelineConfig] = None,
) -> list[ChangeResult]:
    """
    Attach a :class:`~ml_pipeline.models.BoundedSummary` to every
    :class:`~ml_pipeline.models.ChangeResult` in *change_results*.

    Mutates each result's ``summary`` field in-place and returns the list.

    Parameters
    ----------
    change_results:
        Results from :func:`~ml_pipeline.detector.detect_changes`.
    config:
        Pipeline configuration.

    Returns
    -------
    list[ChangeResult]
        The same list with ``summary`` populated on each item.
    """
    if config is None:
        config = load_config()

    for result in change_results:
        try:
            result.summary = get_summary(result, config)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to generate summary for %s: %s",
                result.match.clause_a.clause_id,
                exc,
            )
            result.summary = BoundedSummary(
                summary="Summary generation failed.",
                provider_used="none",
                fallback_reason=str(exc),
            )

    return change_results


# Bring re into module scope for use in GeminiProvider._call closure
import re  # noqa: E402
