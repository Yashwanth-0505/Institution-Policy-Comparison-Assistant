# POLICYX — Institution Policy Comparison Assistant

AI-powered tool that compares two policy PDF documents and identifies every clause change with semantic diff, citations, and plain-English summaries.

**JIG26_25 Hackathon Project**

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env  # Edit with your API keys

# 3. Run the web app
cd frontend
python app.py
```

Open **http://localhost:5000** — the browser launches automatically.

**Admin Dashboard:** Use code `200616`

---

## 📁 Project Structure

```
jig/
├── ml_pipeline/              # Core ML comparison engine
│   ├── pipeline.py           # Main orchestration (8-stage pipeline)
│   ├── extractor.py          # PDF text extraction
│   ├── segmenter.py          # Clause detection
│   ├── features.py           # TF-IDF + embeddings
│   ├── aligner.py            # Baseline & hybrid matchers
│   ├── detector.py           # Change classification
│   ├── summarizer.py         # LLM summary generation
│   ├── citations.py          # Citation attachment
│   ├── evaluation/           # Evaluation framework
│   │   ├── runner.py         # Run benchmarks
│   │   └── data/             # Labelled test pairs
│   └── investigation/        # Missing policy discovery
│       ├── state_machine.py  # Investigation workflow
│       ├── searcher.py       # Web search + scraping
│       ├── provenance.py     # Source verification
│       └── downloader.py     # PDF download
├── frontend/                 # Flask web application
│   ├── app.py                # Main server
│   ├── database.py           # SQLite auth + logging
│   ├── templates/            # HTML pages
│   │   ├── index.html        # Comparison UI
│   │   ├── home.html         # Landing page
│   │   ├── login.html        # Auth page
│   │   └── admin.html        # Admin dashboard
│   └── static/               # CSS styles
├── sample_policies/          # Demo PDFs
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

---

## ✨ Features

### Core Comparison Engine
- **8-Stage ML Pipeline:** Extract → Segment → Features → Align → Detect → Summarize → Cite → Assemble
- **Smart Change Detection:** Distinguishes unchanged / wording-only / substantive / added / removed clauses
- **Semantic Signals:** Detects numeric, date, monetary, negation, modal, and condition changes
- **Hybrid Matching:** Combines TF-IDF cosine + sentence embeddings + positional hints
- **LLM Summaries:** Falls back through Gemini → Groq → Deterministic provider
- **Citation Validation:** Attaches verified page numbers to every change

### Investigation Mode (Missing Previous Policy)
- **Guided Workflow:** User uploads only the newer policy
- **Metadata Extraction:** Extracts institution, year, department from PDF
- **Web Search:** Searches university websites, Google, DuckDuckGo
- **Provenance Verification:** Ranks candidates by domain authority + title match
- **Auto-Download:** Downloads and compares the discovered previous policy

### Web Application
- **Authentication:** User registration + login with session cookies
- **Side-by-Side Comparison:** Semantic diff with color-coded changes
- **Admin Dashboard:** User stats, comparison logs, analytics
- **Sample Demo:** Pre-loaded Greenfield Policy comparison
- **Investigation UI:** Step-by-step guided discovery workflow

---

## 🔧 Configuration

Edit `.env` file:

```env
# Admin Dashboard Access Code
POLICYX_ADMIN_CODE=200616

# Flask Secret Key
POLICYX_SECRET=your-secret-key-here

# LLM API Keys (Optional — local fallback if omitted)
GEMINI_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key

# Pipeline Overrides (Optional)
POLICYX_ACCEPT_THRESHOLD=0.50
POLICYX_REVIEW_THRESHOLD=0.35
POLICYX_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🎯 Usage

### Web Interface

```bash
cd frontend
python app.py
```

### Python API

```python
from ml_pipeline import compare_policies

result = compare_policies(
    policy_a_path="policy_2023.pdf",
    policy_b_path="policy_2024.pdf",
)

# Access results
print(f"Changes detected: {len(result.changes)}")
print(f"Added clauses: {len(result.added_clauses)}")
print(f"Removed clauses: {len(result.removed_clauses)}")

# Export to JSON
json_output = result.model_dump_json(indent=2)
```

### Investigation Mode (Python)

```python
from ml_pipeline.investigation import (
    extract_metadata,
    search_for_previous_policy,
    rank_candidates,
    download_candidate,
)

# Extract metadata from newer policy
meta = extract_metadata("policy_2024.pdf")

# Search for previous version
candidates = search_for_previous_policy(
    institution_name="Greenfield University",
    policy_title="Academic Regulations",
    academic_year="2023",
)

# Rank by provenance
ranked = rank_candidates(candidates, "Greenfield University", ...)

# Download best match
result = download_candidate(ranked[0]["source_url"])
```

---

## 🧪 Run Evaluation

```bash
python -m ml_pipeline.evaluation.runner
```

**Output:**
- Accuracy metrics (precision, recall, F1)
- Wording-only detection test (PASS/FAIL)
- Matcher comparison (baseline vs hybrid)
- Known limitations report

**Dataset:** 30 labelled clause pairs in `ml_pipeline/evaluation/data/`

---

## 📊 Change Types

| Type | Description | Color |
|------|-------------|-------|
| **UNCHANGED** | Identical or near-identical clauses | Green |
| **WORDING_ONLY** | Same meaning, different words | Yellow |
| **SUBSTANTIVE** | Meaning or obligations altered | Red |
| **ADDED** | New clause in policy B | Blue |
| **REMOVED** | Clause deleted from policy A | Pink |
| **UNCERTAIN** | Needs manual review | Purple |

---

## 🔬 Technical Details

### Pipeline Stages

1. **Extraction:** PyMuPDF parsing with OCR warning detection
2. **Segmentation:** Regex-based heading detection + clause splitting
3. **Features:** TF-IDF vectorization + sentence-transformers embeddings
4. **Alignment:** Hybrid matcher (TF-IDF 20% + embedding 50% + position 30%)
5. **Detection:** Signal-based classification (numeric, negation, modal, etc.)
6. **Summarization:** LLM provider chain with 10s timeout
7. **Citations:** Page number validation + exact text snippets
8. **Assembly:** JSON-serializable `ComparisonResult` output

### Wording-Only Detection

Uses a two-stage approach:
1. **difflib ratio >= 0.85** → candidate for wording-only
2. **No semantic signals** (numeric, date, negation, modal) → confirmed

Any semantic signal → classified as **SUBSTANTIVE** regardless of similarity.

### Investigation State Machine

```
IDLE → COLLECTING_CONTEXT → EXTRACTING_METADATA → SEARCHING_SOURCES
→ RANKING_CANDIDATES → VERIFYING_PROVENANCE → AWAITING_CONFIRMATION
→ DOWNLOADING_DOCUMENT → COMPARING_POLICIES → COMPLETED
```

Each state transition is logged and can be inspected via the web UI.

---

## 🧑‍💻 Development

### Run Tests

```bash
pytest ml_pipeline/tests/
```

### Debug Scripts

```bash
python debug_searcher.py      # Test web search
python debug_signals.py       # Test signal detection
python debug_wording_only.py  # Test wording-only classifier
python test_investigation.py  # Test investigation workflow
```

### Create Sample PDFs

```bash
python create_sample_pdfs.py
```

Generates synthetic policy PDFs in `sample_policies/`

---

## 📦 Dependencies

- **pymupdf** — PDF text extraction
- **sentence-transformers** — Semantic embeddings
- **scikit-learn** — TF-IDF + cosine similarity
- **rapidfuzz** — Fast string matching
- **pydantic** — Data validation + JSON serialization
- **flask** — Web server
- **google-genai** — Gemini LLM API
- **openai** — Groq LLM API (OpenAI-compatible)

Full list in `requirements.txt` (Python 3.11+ required)

---

## 🎓 Use Cases

- **University Policy Updates:** Compare academic regulations year-over-year
- **Compliance Audits:** Track substantive vs wording-only changes
- **Legal Review:** Identify obligation shifts in contracts
- **Version Control:** Semantic diff for non-code documents
- **Regulatory Analysis:** Monitor policy evolution over time

---

## ⚠️ Known Limitations

- **PDF Quality:** Scanned/image PDFs produce OCR warnings; use searchable PDFs
- **Heading Detection:** Custom heading styles may not be recognized
- **Embedding Model:** Requires offline cache or internet for first run
- **Small Dataset:** 30 labelled pairs insufficient for statistical robustness
- **LLM Hallucination:** Deterministic fallback always available
- **No Multi-Doc:** Compares exactly two documents per call
- **Memory-Only Matching:** No vector database; O(n²) scaling

---

## 🏆 Hackathon Deliverables

✅ **Day 1:** Core pipeline (8 stages) + wording-only detection  
✅ **Day 2:** Investigation mode (missing policy discovery)  
✅ **Day 3:** Web UI + auth + admin dashboard  
✅ **Evaluation:** 30 labelled pairs + automated test suite  
✅ **Demo:** Greenfield sample policies pre-loaded  

**Admin Code:** `200616`

---

## 📝 License

MIT License — Free for educational and commercial use.

---

## 👥 Team

**JIG26_25** — Institution Policy Comparison Assistant

Built for the 2026 Hackathon 🚀

---

## 🔗 Quick Links

- **Live Demo:** http://localhost:5000
- **Admin Dashboard:** http://localhost:5000/admin (code: 200616)
- **Sample Comparison:** http://localhost:5000/sample
- **ML Pipeline Docs:** `ml_pipeline/README.md`
- **Evaluation Data:** `ml_pipeline/evaluation/data/`

---

**Questions?** Check the inline documentation in each module or run:

```bash
python -c "from ml_pipeline import compare_policies; help(compare_policies)"
```
