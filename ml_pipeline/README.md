# POLICYX ML Pipeline

**Institution Policy Comparison Assistant** — JIG26_25

Compares two policy PDF documents and returns a structured JSON report of every clause change: unchanged, wording-only rewordings, substantive changes, added clauses, and removed clauses — each with cited sources and a plain-English summary.

---

## 1. Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ required. All dependencies are pinned in `requirements.txt`.

---

## 2. Pre-download the Embedding Model (offline cache)

Run this **before** the hackathon to cache the model locally so it works without internet access:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Model is now cached in ~/.cache/huggingface/
```

Or via CLI:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

---

## 3. Environment Variables

Create a `.env` file in the project root (parent of `ml_pipeline/`):

```dotenv
# LLM providers (optional — pipeline falls back to deterministic if absent)
GEMINI_API_KEY=your-gemini-api-key-here
GROQ_API_KEY=your-groq-api-key-here

# Override pipeline defaults (all optional)
POLICYX_ACCEPT_THRESHOLD=0.50
POLICYX_REVIEW_THRESHOLD=0.35
POLICYX_NUMBER_WEIGHT=0.20
POLICYX_HEADING_WEIGHT=0.30
POLICYX_EMBEDDING_WEIGHT=0.50
POLICYX_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
POLICYX_GEMINI_MODEL=gemini-2.5-flash
POLICYX_GROQ_MODEL=llama-3.3-70b-versatile
POLICYX_SUMMARY_TIMEOUT=10
POLICYX_MIN_CLAUSE_LENGTH=20
```

API keys are **never hardcoded** — always read from the environment.

---

## 4. Compare Two Policies

```python
from ml_pipeline import compare_policies

result = compare_policies(
    policy_a_path="path/to/policy_v1.pdf",
    policy_b_path="path/to/policy_v2.pdf",
)

# Access results
print(f"Processing complete: {result.processing_complete}")
print(f"Changes detected:    {len(result.changes)}")
print(f"Added clauses:       {len(result.added_clauses)}")
print(f"Removed clauses:     {len(result.removed_clauses)}")

# Inspect each change
for change in result.changes:
    print(f"  [{change.change_type.value}] {change.match.clause_a.heading}")
    if change.summary:
        print(f"    Summary: {change.summary.summary}")
    for citation in change.citations:
        print(f"    Cited: {citation.document_name}, p.{citation.page_number}")

# Serialise to JSON
json_output = result.model_dump_json(indent=2)
```

---

## 5. Run Evaluation

```bash
# From the d:\jig directory (parent of ml_pipeline/)
python -m ml_pipeline.evaluation.runner
```

Example output:

```
============================================================
POLICYX ML Pipeline — Evaluation Report
============================================================
Timestamp : 2026-10-03T10:30:00+00:00
Dataset   : 30 labelled pairs
Correctly matched (hybrid): 24 / 30 (accuracy: 80.0%)

── Wording-Only Detection (Day 2 Sprint) ──────────────
  Total wording-only pairs in dataset : 6
  Correctly labelled as wording_only  : 6
  Incorrectly labelled as substantive : 0
  Result : ✅ PASS

── Matcher Comparison ──────────────────────────────────
  Metric                          Baseline      Hybrid
  ────────────────────────────────────────────────────
  Precision                         0.7500      0.8500
  Recall                            0.7200      0.8000
  F1 (alignment)                    0.7347      0.8242
  Macro F1 (classification)         0.6200      0.7100

── Known Limitations ───────────────────────────────────
  • Evaluation uses synthetic clause pairs, not real PDF documents.
  • Embedding model may not be cached offline; embeddings fall back to TF-IDF.
  • LLM summaries are not evaluated (deterministic provider used in eval).
  • Wording-only detection relies on difflib ratio; domain-specific paraphrases
    may score below threshold.
  • Dataset size (30 pairs) is small; results may not generalise to all policy
    types.
============================================================
```

---

## 6. LLM Fallback Chain

The pipeline tries three summary providers in order:

```
GeminiProvider  →  GroqProvider  →  DeterministicProvider
```

| Provider | Requires | Behaviour on failure |
|---|---|---|
| **GeminiProvider** | `GEMINI_API_KEY` env var | Raises → next provider tried |
| **GroqProvider** | `GROQ_API_KEY` env var | Raises → next provider tried |
| **DeterministicProvider** | Nothing | Always succeeds — no API |

- If a key is missing, `is_available()` returns `False` and the provider is skipped (not tried).
- If a key is present but the API call fails (timeout, rate-limit, bad response), the exception is caught, logged, and the next provider is tried.
- The `BoundedSummary.provider_used` field records which provider succeeded.
- The `BoundedSummary.fallback_reason` field records why earlier providers were skipped.
- The LLM prompt sends **only** the two clause texts and detected signals — it is explicitly instructed not to invent citations or legal consequences.

---

## 7. Day 2 Sprint Compliance — Wording-Only Detection

A core requirement of POLICYX is distinguishing **wording-only changes** (same meaning, different words) from **substantive changes** (meaning altered).

### How it works

The detector uses a two-stage approach:

1. **difflib.SequenceMatcher ratio** — measures character-level similarity.
   - `ratio >= 0.98` → UNCHANGED
   - `ratio >= 0.85` with no semantic signals → WORDING_ONLY
   - `ratio < 0.60` → UNCERTAIN

2. **Semantic signal extraction** — looks for:
   - Numeric changes (`30 days` → `60 days`)
   - Date changes
   - Monetary changes
   - Negation word additions/removals (`shall` → `shall not`)
   - Modal obligation shifts (`may` → `must`)
   - Condition phrase changes (`if`, `unless`, `provided that`)

   Any of the above signals → SUBSTANTIVE (regardless of ratio)

### Evaluation result

The dataset contains 6 wording-only pairs (3 `paraphrased_wording_only` + 3 `renumbered_same_meaning`). The evaluation report's `wording_only_test` block confirms:

- How many were correctly classified as `wording_only`
- How many were **incorrectly** classified as `substantive` (must be 0 to PASS)
- Overall PASS/FAIL status

Run `python -m ml_pipeline.evaluation.runner` to see the live result.

---

## 8. JSON Output Schema Overview

`compare_policies()` returns a `ComparisonResult` whose `.model_dump_json()` produces:

```json
{
  "document_a": { "document_id": "doc_a", "document_name": "...", "pages": [...] },
  "document_b": { "document_id": "doc_b", "document_name": "...", "pages": [...] },
  "clauses_a": [ { "clause_id": "doc_a_clause_0000", "heading": "...", ... } ],
  "clauses_b": [ ... ],
  "matches": [ { "clause_a": {...}, "clause_b": {...}, "match_score": 0.91, "match_type": "hybrid" } ],
  "added_clauses": [ { "clause_id": "doc_b_clause_0005", ... } ],
  "removed_clauses": [ { "clause_id": "doc_a_clause_0003", ... } ],
  "changes": [
    {
      "match": { "clause_a": {...}, "clause_b": {...}, "match_type": "hybrid" },
      "change_type": "substantive",
      "signals": [
        { "signal_type": "numeric_change", "description": "...", "old_value": "30", "new_value": "60" }
      ],
      "needs_review": false,
      "summary": {
        "summary": "The notice period has been doubled from 30 to 60 days.",
        "classification_suggestion": "substantive",
        "confidence": 0.9,
        "provider_used": "gemini",
        "fallback_reason": ""
      },
      "citations": [
        { "document_id": "doc_a", "clause_id": "doc_a_clause_0002", "page_number": 3,
          "exact_text": "...", "citation_valid": true }
      ]
    }
  ],
  "warnings": [],
  "errors": [],
  "processing_complete": true
}
```

---

## 9. Backend Integration (FastAPI)

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os
from ml_pipeline import compare_policies

app = FastAPI(title="POLICYX API")

@app.post("/compare")
async def compare(
    policy_a: UploadFile = File(...),
    policy_b: UploadFile = File(...),
):
    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "policy_a.pdf")
        path_b = os.path.join(tmp, "policy_b.pdf")
        with open(path_a, "wb") as f:
            f.write(await policy_a.read())
        with open(path_b, "wb") as f:
            f.write(await policy_b.read())
        try:
            result = compare_policies(path_a, path_b)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(content=result.model_dump())
```

Start the server:

```bash
uvicorn main:app --reload
```

---

## 10. Known Limitations

- **PDF quality**: Scanned/image-only pages produce an OCR warning and may yield empty text. Use searchable PDFs for best results.
- **Clause segmentation**: The heading-detection regex covers common policy formats but may miss custom heading styles. Unrecognised text lands in a single preamble clause flagged for review.
- **Embedding model offline**: If `sentence-transformers/all-MiniLM-L6-v2` is not cached and there is no internet, embeddings fall back to TF-IDF cosine silently.
- **Evaluation dataset size**: 30 labelled pairs is sufficient for demonstration but too small to draw statistically robust conclusions about real-world accuracy.
- **Wording-only threshold sensitivity**: The `ratio >= 0.85` threshold works well for near-identical paraphrases but may misclassify highly restructured same-meaning clauses as UNCERTAIN. Tune `POLICYX_ACCEPT_THRESHOLD` if needed.
- **LLM hallucination**: LLM summaries are instructed not to invent information, but this cannot be fully guaranteed. The deterministic fallback is always available and produces factual, signal-based summaries.
- **No multi-document comparison**: The pipeline compares exactly two documents per call.
- **No vector database**: All matching is done in-memory; performance scales roughly as O(n²) in clause count.
