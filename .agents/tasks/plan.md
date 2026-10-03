# Implementation Plan: Missing Previous Policy Investigation Mode

This plan implements a new investigation workflow for POLICYX that allows users to upload only the newer policy document, answer 2 context questions, search for the previous version from authoritative sources, verify provenance, and feed the discovered document into the existing ML comparison pipeline.

**Design Decisions:**

1. **Search Provider: Serper API** — Chosen for best free tier (2,500 free searches/month) and access to Google's index for institutional documents. Alternatives: Brave ($5/1k, no free tier), DuckDuckGo (no official API). Serper provides reliable .edu domain discovery and supports filetype:pdf filtering.

2. **State Machine Architecture** — Investigation flow modeled as explicit state machine (10 states) to ensure predictable transitions, enable resume capability, and simplify error handling. States stored in InvestigationSession model.

3. **Provenance Verification Algorithm** — Multi-signal scoring (7 signals: domain authority, institution/university/title consistency, version/date/authority consistency) with 4-tier status (VERIFIED ≥20 pts, STRONG ≥15, LIMITED ≥8, UNVERIFIED <8). Prioritizes official university domains (.edu) and exact name matches.

4. **API Endpoint Design** — 6 REST endpoints (/investigate/start, /context, /search, /confirm, /status, optional /cancel) model each state transition. Stateless HTTP design with session stored server-side (in-memory for demo; production would use Redis).

5. **Frontend Integration** — Mode toggle (radio buttons) switches between existing two-PDF comparison and new investigation mode. Investigation reuses existing upload box styling and comparison results UI, adding dialog for 2 questions and candidate selection list. Zero changes to existing /compare logic.

6. **Separation of Concerns** — Investigation module (ml_pipeline/investigation/) is independent of core pipeline. compare_policies() unchanged. Discovered PDF treated as untrusted input, fed through same extraction/segmentation/matching pipeline as user uploads.

---

## Feature Decomposition

This task decomposes into 4 features implemented sequentially:

- **FEAT-001**: Investigation module (backend logic, models, state machine, search client, provenance verifier)
- **FEAT-002**: Flask API endpoints (6 routes integrating investigation module with existing app)
- **FEAT-003**: Frontend UI (mode toggle, 2-question dialog, candidate selection, provenance display)
- **FEAT-004**: Comprehensive tests (unit, integration, isolation, edge cases)

See `d:\jig\.agents\tasks\task-investigation-mode\features\*.json` for detailed FEAT specifications.

---

## Fallback: Traditional Plan (if FEAT decomposition not used)

- [ ] 1. Create investigation module structure and models
      Create ml_pipeline/investigation/ directory with __init__.py, models.py (InvestigationState enum, InvestigationContext, CandidateDocument, InvestigationSession Pydantic models), state_machine.py (InvestigationStateMachine class with transition validation).
      Files: ml_pipeline/investigation/__init__.py, ml_pipeline/investigation/models.py, ml_pipeline/investigation/state_machine.py
      Verify: Import investigation models in Python REPL: `from ml_pipeline.investigation.models import InvestigationState, InvestigationContext, CandidateDocument` — no import errors

- [ ] 2. Implement Serper API search client
      Create ml_pipeline/investigation/search_client.py with SerperSearchClient class. Implements search_policies(institution, university, policy_title) method using Serper API (https://google.serper.dev/search) with query: "{institution} {university} {policy_title} policy filetype:pdf". Parse organic results for title, link, snippet, domain.
      Files: ml_pipeline/investigation/search_client.py
      Verify: Unit test with mocked requests.post response; verify query construction and response parsing

- [ ] 3. Implement metadata extraction and provenance verification
      Create ml_pipeline/investigation/metadata_extractor.py (extract_metadata_from_pdf using PyMuPDF to find title, institution, dates, version) and ml_pipeline/investigation/provenance_verifier.py (ProvenanceVerifier class with 7-signal scoring algorithm: domain_authority, institution/university/title/version/date/authority consistency). Implement 4-tier provenance status thresholds.
      Files: ml_pipeline/investigation/metadata_extractor.py, ml_pipeline/investigation/provenance_verifier.py
      Verify: Unit test ProvenanceVerifier with mock CandidateDocument objects covering all signals; verify VERIFIED (≥20), STRONG (≥15), LIMITED (≥8), UNVERIFIED (<8) thresholds

- [ ] 4. Implement candidate ranking
      Create ml_pipeline/investigation/ranker.py with rank_candidates(candidates, context) function. Sort by: 1) provenance_status (VERIFIED > STRONG > LIMITED > UNVERIFIED), 2) confidence_score descending, 3) source_domain priority (official university > institution > .edu > .gov > other).
      Files: ml_pipeline/investigation/ranker.py
      Verify: Unit test with 5 mock candidates having different statuses and scores; verify sort order matches specification

- [ ] 5. Update pipeline config for API keys
      Add serper_api_key and brave_api_key fields to PipelineConfig in ml_pipeline/config.py. Update load_config() to read SERPER_API_KEY and BRAVE_API_KEY from environment via python-dotenv.
      Files: ml_pipeline/config.py
      Verify: `from ml_pipeline.config import load_config; cfg = load_config(); print(cfg.serper_api_key)` — prints empty string if not set, key value if SERPER_API_KEY in .env

- [ ] 6. Add investigation Flask endpoints to frontend/app.py
      Add 6 new routes: /investigate/start (POST, accepts newer_policy file, returns session_id), /investigate/context (POST, accepts 2-question form data, returns state), /investigate/search (POST, triggers search and provenance verification, returns candidates), /investigate/confirm (POST, accepts candidate_index, downloads PDF, runs compare_policies, returns comparison result), /investigate/status/<id> (GET, returns session state), optional /investigate/cancel. Add global investigation_sessions dict for in-memory session storage. All endpoints validate state transitions using InvestigationStateMachine.
      Files: frontend/app.py
      Verify: Start Flask server `python frontend/app.py`, test /investigate/start with curl: `curl -F 'newer_policy=@sample_policies/Greenfield_Policy_2024.pdf' http://localhost:5000/investigate/start` — returns JSON with session_id and state='COLLECTING_CONTEXT'

- [ ] 7. Add investigation mode toggle and dialog UI to index.html
      Add radio button group for mode selection ('Compare two versions' vs 'Find earlier version') in upload-workspace section. Add hidden investigation-dialog section with 2-question form (institution_name, affiliating_university, policy_title, optional department, optional academic_year). Add hidden candidate-list section. JavaScript: mode toggle shows/hides uploadA box, changes button text. In investigate mode, upload triggers /investigate/start → show dialog → on submit POST /investigate/context → POST /investigate/search → render candidates.
      Files: frontend/templates/index.html
      Verify: Open http://localhost:5000, switch to 'Find earlier version' mode, verify uploadA hidden and button text changed. Upload PDF, verify dialog appears.

- [ ] 8. Add candidate selection and provenance display UI
      JavaScript renderCandidates() function creates candidate-item divs showing title, institution/university, source domain, provenance badge (VERIFIED=green, STRONG=blue, LIMITED=yellow, UNVERIFIED=red), provenance signals summary. On candidate click, call confirmCandidate(index) → POST /investigate/confirm → load comparison results using existing load() function. Add investigation metadata to comparison header (discovered policy source + provenance badge).
      Files: frontend/templates/index.html
      Verify: In investigate mode, after search returns candidates, verify candidate list displays with provenance badges. Click candidate, verify /investigate/confirm called and comparison results load in existing UI.

- [ ] 9. Add investigation CSS styles to style.css
      Add styles for: .mode-toggle (radio button group), .investigation-dialog (modal overlay), .dialog-content (form container), .candidate-list, .candidate-item (hover effects), .provenance-badge (4 variants: verified/strong/limited/unverified colors), .provenance-signals (metadata text). Follow existing design system (CSS custom properties, same colors/typography/spacing/shadows).
      Files: frontend/static/style.css
      Verify: Visual inspection in browser: dialog and candidate list match existing UI style. Provenance badges use correct colors (green/blue/yellow/red). Hover states work on candidate items.

- [ ] 10. Add investigation module unit tests
      Create ml_pipeline/investigation/tests/ with test_state_machine.py (valid/invalid transitions), test_provenance_verifier.py (scoring algorithm for all 4 statuses), test_search_client.py (mocked Serper API), test_metadata_extractor.py (PDF parsing), test_ranker.py (sort order), conftest.py (fixtures: make_investigation_context, make_candidate_document, mock_serper_response).
      Files: ml_pipeline/investigation/tests/test_*.py, ml_pipeline/investigation/tests/conftest.py
      Verify: `pytest ml_pipeline/investigation/tests/ -v` from d:\jig — all tests pass

- [ ] 11. Add investigation integration and isolation tests
      Create test_investigation_endpoints.py (Flask test client tests for all 6 routes, state transitions, error handling) and test_investigation_isolation.py (verify /compare and /sample endpoints unchanged, concurrent investigation + comparison). Use mocked SerperSearchClient and requests.get for PDF download. Test missing SERPER_API_KEY returns 400. Test invalid session_id returns 404.
      Files: test_investigation_endpoints.py, test_investigation_isolation.py
      Verify: `pytest test_investigation_endpoints.py test_investigation_isolation.py -v` — all tests pass. `pytest ml_pipeline/tests/test_pipeline.py -v` — existing tests still pass (no regression).

- [ ] 12. Update requirements.txt with new dependencies
      Add requests>=2.31.0 (for PDF download in /investigate/confirm), rapidfuzz>=3.9.0 (already present, used for provenance title matching). No other dependencies needed (PyMuPDF, python-dotenv, Flask already present).
      Files: requirements.txt
      Verify: `pip install -r requirements.txt` completes without errors

- [ ] 13. Create .env.example with investigation API keys
      Create .env.example file documenting SERPER_API_KEY and BRAVE_API_KEY (optional). Include comment: "Get free Serper API key (2,500 searches/month) at https://serper.dev/". Add note that investigation mode requires SERPER_API_KEY; compare mode works without it.
      Files: .env.example
      Verify: File exists with clear documentation of required vs optional keys

- [ ] 14. End-to-end manual verification
      Start Flask server, test full investigation flow: 1) Switch to investigate mode, 2) Upload sample_policies/Greenfield_Policy_2024.pdf, 3) Fill 2-question form with 'Greenfield University' / 'Academic Integrity Policy', 4) Verify candidates appear (if SERPER_API_KEY set), 5) Select candidate, verify comparison loads. Test existing compare mode: upload 2 PDFs, verify comparison unchanged. Test manual upload option if no candidates found.
      Files: N/A (manual test)
      Verify: Investigation flow completes successfully. Existing comparison flow unchanged. No console errors. Provenance badges display correctly.

---

## API Endpoint Specifications

### POST /investigate/start
**Request**: `multipart/form-data` with `newer_policy` file  
**Response**: `{session_id: str, state: str, message: str}`  
**State transition**: None → COLLECTING_CONTEXT

### POST /investigate/context
**Request**: `{session_id: str, institution_name: str, affiliating_university: str, policy_title: str, department?: str, academic_year?: str}`  
**Response**: `{session_id: str, state: str, message: str}`  
**State transition**: COLLECTING_CONTEXT → EXTRACTING_METADATA → SEARCHING_SOURCES

### POST /investigate/search
**Request**: `{session_id: str}`  
**Response**: `{session_id: str, state: str, candidates: [{url, title, institution_name, university_name, policy_title, version, source_domain, provenance_status, confidence_score, provenance_signals: {domain_authority, institution_match, university_match, title_match, version_check, date_check, authority_check}}]}`  
**State transition**: SEARCHING_SOURCES → RANKING_CANDIDATES → VERIFYING_PROVENANCE → AWAITING_CONFIRMATION

### POST /investigate/confirm
**Request**: `{session_id: str, candidate_index: int}`  
**Response**: ComparisonResult JSON (matching /compare format) + `{investigation: {session_id, discovered_policy: {url, title, provenance_status, provenance_signals}, uploaded_policy: {filename, metadata}}}`  
**State transition**: AWAITING_CONFIRMATION → DOWNLOADING_DOCUMENT → COMPARING_POLICIES → COMPLETED

### GET /investigate/status/<session_id>
**Response**: `{session_id: str, state: str, context: {...}, candidates?: [...], errors?: [...]}`

---

## Provenance Scoring Algorithm

| Signal | Points | Criteria |
|--------|--------|----------|
| Domain authority | 5 | Official .edu university domain |
| | 3 | Official college/institution domain |
| | 1 | Other domains |
| Institution name consistency | 4 | Exact match (case-insensitive) |
| | 2 | Partial match (fuzzy ≥70%) |
| University consistency | 4 | Exact or partial match |
| Policy title consistency | 5 | Fuzzy ratio ≥85% |
| | 3 | Fuzzy ratio ≥70% |
| | 1 | Fuzzy ratio ≥50% |
| Version consistency | 3 | Candidate version < newer version |
| Date consistency | 3 | Candidate date < newer date |
| Issuing authority consistency | 2 | Same department/office |

**Provenance Status Thresholds:**
- **VERIFIED_PROVENANCE**: ≥20 points
- **STRONG_PROVENANCE**: ≥15 points
- **LIMITED_PROVENANCE**: ≥8 points
- **UNVERIFIED**: <8 points

---

## Investigation States

1. **IDLE** — No active investigation
2. **COLLECTING_CONTEXT** — Awaiting 2-question form submission
3. **EXTRACTING_METADATA** — Parsing uploaded PDF metadata
4. **SEARCHING_SOURCES** — Querying Serper API
5. **RANKING_CANDIDATES** — Sorting search results
6. **VERIFYING_PROVENANCE** — Scoring each candidate
7. **AWAITING_CONFIRMATION** — User selecting candidate
8. **DOWNLOADING_DOCUMENT** — Fetching confirmed candidate PDF
9. **COMPARING_POLICIES** — Running compare_policies()
10. **COMPLETED** — Comparison result returned
11. **FAILED** — Unrecoverable error occurred

---

## Source Priority Ordering

1. Official university domain (e.g., university.edu/policies/)
2. Official college/institution domain (e.g., college.edu/)
3. Official institutional repository (e.g., repository.university.edu/)
4. Official government/accreditation source (e.g., .gov, accreditor.org/)
5. Other sources (discovery only, provenance UNVERIFIED)

---

## User Confirmation UI

Each candidate displays:
- **Title**: Extracted document title
- **Institution**: Institution name found in document
- **University**: Affiliated university name
- **Policy Title**: Matched policy title
- **Version**: Document version/date if found
- **Source URL**: Domain and path
- **Provenance Status**: Badge (green/blue/yellow/red)
- **Provenance Signals**: Breakdown (e.g., "Domain: Official university ✓ • Title match: 92% ✓ • Date: Confirmed earlier ✓")

**Actions**: [Use this policy] [Choose another] [Upload manually]

---

## Critical Constraints Verification

✅ **DO NOT break existing two-PDF comparison workflow** — Mode toggle isolates investigate vs compare; /compare endpoint unchanged  
✅ **DO NOT modify ML pipeline alignment/detection logic** — compare_policies() called unchanged in /investigate/confirm  
✅ **DO NOT bypass existing comparison pipeline** — Discovered PDF fed through same pipeline as user uploads  
✅ **DO NOT claim authenticity without verification evidence** — Provenance status + signals displayed; UNVERIFIED clearly labeled  
✅ **LLM generates bounded summaries only** — Investigation uses existing summarizer; no new LLM calls  
✅ **Treat discovered documents as untrusted input** — Downloaded PDF parsed via extractor like any upload  
✅ **Keep API keys in environment variables** — SERPER_API_KEY read via load_config() from .env

---

## Testing Strategy

1. **Unit Tests** (ml_pipeline/investigation/tests/)
   - State machine transitions (valid/invalid)
   - Provenance scoring algorithm (all 4 statuses)
   - Search client query construction and parsing
   - Metadata extraction from PDF
   - Candidate ranking sort order

2. **Integration Tests** (root test files)
   - All 6 Flask endpoints with test client
   - State transitions across full workflow
   - Error handling (missing API key, invalid session, network timeout)
   - Mocked external dependencies (Serper API, PDF download)

3. **Isolation Tests**
   - /compare endpoint unchanged after investigation code added
   - /sample endpoint unchanged
   - Concurrent investigation + comparison requests
   - investigation_sessions dict isolation

4. **Edge Cases**
   - No candidates found (empty search results)
   - Candidate with no institution name
   - Candidate from wrong university
   - All signals matching (VERIFIED)
   - Partial signals matching (STRONG/LIMITED)
   - PDF download failure
   - Malformed PDF

5. **Manual End-to-End**
   - Full investigation flow with real SERPER_API_KEY
   - Provenance badge colors and signals display
   - Manual upload fallback
   - Existing compare mode verification

---

## Notes

- **Session Storage**: In-memory dict for demo; production should use Redis or database with TTL
- **Cleanup**: Temp files (uploaded + downloaded PDFs) deleted after COMPLETED or FAILED state
- **Rate Limiting**: Serper free tier is 2,500 searches/month; production should implement rate limiting
- **Error Boundaries**: All Flask endpoints wrapped in try/except; errors stored in session.errors
- **Accessibility**: Investigation dialog keyboard-navigable; ARIA labels on form fields and candidate list
- **Responsive**: Investigation UI follows existing responsive patterns (mobile-friendly)
