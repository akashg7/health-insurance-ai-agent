# Master Bug Audit & Fix Plan

This document catalogs every verified bug, architectural flaw, UI/UX issue, and testing/evaluation gap found across the Health Insurance AI Agent codebase, along with a step-by-step plan to fix and verify them one by one.

---

## Part 1: Root-Cause Bugs Found

### A. Critical Correctness & Hallucination Bugs (Why random PDFs say "knee replacement is covered")

1. **Prompt Explicitly Tells LLM to Assume `covered: true` on Missing Evidence**
   - **Location:** `health_insurance_bot/rules_extractor.py` (line 63)
   - **Bug:** The extraction prompt states:
     > *"If excerpts are insufficient to decide, set covered: true, waiting_period_days: 30, and explain uncertainty in notes."*
   - **Impact:** Uploading a random PDF (resume, invoice, brochure, unrelated document) or querying a condition absent from the PDF causes the extractor to fabricate `"covered": true` instead of `"covered": null` / `"not_mentioned"`.

2. **Exception Fallback Silently Marks Failed API Calls as `covered: true`**
   - **Location:** `health_insurance_bot/rules_extractor.py` (lines 88–97)
   - **Bug:** Any exception (Groq HTTP 429 rate limits, network errors, malformed JSON) is caught and returns:
     ```python
     {"covered": True, "sub_limit_inr": None, "co_pay_percent": 0, "waiting_period_days": 0, "page_reference": 0, "notes": f"extraction error: {e}"}
     ```
   - **Verified Evidence in Repo:**
     - `health_insurance_bot/policies/3373ed3d3473bfe1/rules.json` (`Brochure_Super Star.pdf`): All non-exclusion conditions (including `knee replacement`, `cataract surgery`, `hernia repair`) failed with Groq 429 daily token limit (`TPD`) and were permanently saved as `"covered": true`.
     - `health_insurance_bot/policies/91cba41dabc56a27/rules.json` (`Activ_health_addons.pdf`): Failed with Groq 429 `TPD` and saved `"covered": true`.
     - `health_insurance_bot/policies/79662228bf4a3046/rules.json` (`policy.pdf`, HDFC Optima sample): **8 of 48 conditions** (`asthma`, `PCOD`, `GERD`, `varicose veins`, `ICU charges`, `ambulance road`, `AYUSH`, `post hospitalization expenses`) actually failed extraction with Groq 429 `TPM` errors and silently cached `"covered": true` with `waiting_period_days: 0`!

3. **Poisoned `rules.json` Files Are Permanently Cached as Valid**
   - **Location:** `health_insurance_bot/rules_extractor.py` (lines 139–143)
   - **Bug:** `if rules_path.exists() and not force:` skips extraction without checking if the cached file contains `"extraction error"` entries or failed rules. Once a rate-limited extraction happens, that broken JSON is served forever.

4. **No Validation That Uploaded PDF Is a Valid Health Insurance Policy (or Readable Text)**
   - **Location:** `health_insurance_bot/ingest.py` (lines 55–122), `health_insurance_bot/policy_processor.py` (lines 20–35)
   - **Bug:**
     - There is no domain classifier/gatekeeper to check whether the uploaded PDF is actually a health insurance policy document vs. a random document (resume, book, receipt, empty form).
     - There is no handling for scanned/image-only PDFs where `pdfplumber` extracts 0 text (`chunks == []`), which causes `coll.add()` in ChromaDB to crash.

5. **Over-Aggressive Keyword Normalization Overrides Actual PDF Content**
   - **Location:** `health_insurance_bot/rules_extractor.py` (lines 100–132)
   - **Bug:** `normalize_rule()` forces `covered: false` for keywords like `maternity`, `opd`, `routine dental`, `lasik`, `hearing aids` regardless of whether an uploaded policy or rider/add-on PDF actually covers them (or failed extraction, or isn't an insurance policy at all). It also flips `covered: false` to `true` whenever `waiting_period_days > 0`—which combined with Bug #1 (`waiting_period_days: 30` on insufficient excerpts) guarantees false positives.

6. **Broken Distance Metric & Missing Relevance Threshold in Vector Search**
   - **Location:** `health_insurance_bot/ingest.py` (lines 89, 125–137), `health_insurance_bot/tools/policy_search.py` (line 26)
   - **Bug:**
     - `client.create_collection(name)` defaults to squared L2 distance (`0.0` to `4.0`), NOT cosine distance.
     - `policy_search.py` computes `1 - h["distance"]`, producing negative/nonsensical relevance scores for L2 distances `> 1.0`.
     - Neither `search_policy_docs` nor `rules_extractor` filters out irrelevant low-similarity chunks (no similarity cutoff), and `coll.query(..., n_results=k)` fails if `coll.count() < k` (small PDFs).

7. **Broken Parallel Extraction Strategy Blows Groq Rate Limits**
   - **Location:** `health_insurance_bot/rules_extractor.py` (lines 24–25, 161)
   - **Bug:** Fires 48 individual `llama-3.3-70b-versatile` calls across 5 worker threads with zero rate limiting, batching, or exponential backoff, immediately hitting Groq's 12k TPM / 100k TPD caps. Also does not share Streamlit secrets fallback cleanly when invoked cleanly.

---

### B. Agent, Tool & State Management Bugs

8. **Fuzzy Condition Matcher Produces False Positives (`"dental surgery"` -> `"cataract surgery"`)**
   - **Location:** `health_insurance_bot/tools/coverage.py` (lines 9–30, 48–55)
   - **Bug:**
     - `_find_condition()` uses `best_score >= 1` single-token overlap. Generic words like `"surgery"`, `"treatment"`, `"disease"`, `"hospitalization"`, or `"expenses"` match completely unrelated conditions in insertion order!
     - `check_coverage()` returns the user's input string as `"condition"` rather than the matched rule key, hides extraction errors (`page_reference == 0`), and makes the LLM present unverified defaults as policy facts.

9. **Agent Has Zero Conversational Memory (`chat_history` Never Passed)**
   - **Location:** `health_insurance_bot/agent.py` (line 75) vs. `health_insurance_bot/app.py` (line 590)
   - **Bug:** `agent.py` defines `MessagesPlaceholder("chat_history", optional=True)`, but `app.py` calls `ss.agent.invoke({"input": user_msg})` without passing `chat_history`. Multi-turn follow-up questions (*"How much will I get for a ₹50,000 bill?"* after *"Is cataract covered?"*) lose all prior context.

10. **System Prompt Forces Confident Citations Even When Evidence Is Missing**
    - **Location:** `health_insurance_bot/agent.py` (lines 18–49)
    - **Bug:** Prompt instructs the model to always *"Lead with a 1-sentence direct answer in bold"* and *"End with a one-line citation: Source: policy page X"*, without explicitly checking if the loaded document failed validation, had extraction errors, or lacked evidence (`covered=None`, `status="not_mentioned"`).

11. **`calculate_claim()` Ignores Waiting Periods, Extraction Status, and User Override**
    - **Location:** `health_insurance_bot/tools/coverage.py` (lines 59–111)
    - **Bug:**
      - Does not check whether the user's policy tenure (`policy_start` vs today) has cleared the condition's `waiting_period_days` (even though `get_user_policy`'s docstring explicitly claims this!).
      - Does not account for or warn about waiting period / extraction errors / missing evidence (`page_reference == 0`).
      - Uses global `CTX.user_id` even if the LLM specifically queried a different user profile ID.

12. **Process-Global Mutable State (`CTX`) Causes Cross-Session Race Conditions**
    - **Location:** `health_insurance_bot/tools/_context.py` (lines 16–46)
    - **Bug:** `CTX` is a module-level singleton shared across all concurrent browser sessions in the same Streamlit server. Two simultaneous users overwrite each other's active PDF policy and user profile.

13. **Mock Static Data Mixed Silently With Custom Uploaded PDFs**
    - **Location:** `health_insurance_bot/tools/hospitals.py`, `health_insurance_bot/tools/documents.py`, `health_insurance_bot/tools/user.py`
    - **Bug:** Regardless of which insurer's PDF is uploaded, hospital lookup and document checklists read static mock files without indicating to the user/agent that network hospital data is sample directory data rather than extracted from their PDF.

---

### C. UI / UX & Session Bugs (`app.py`)

14. **"Load Sample Policy" Button Gets Immediately Overwritten by `st.file_uploader`**
    - **Location:** `health_insurance_bot/app.py` (lines 435–446)
    - **Bug:** Once a user uploads a PDF, `uploaded` remains populated inside `st.file_uploader`. Clicking *"Load Sample (HDFC Optima)"* switches the active policy momentarily, then on the very next chat rerun `if uploaded is not None:` sees `pdf_fingerprint(tmp) != ss.processed_pdf_hash` and switches right back to the uploaded PDF!

15. **Custom Profile State Gets Lost/Overwritten Across Reruns**
    - **Location:** `health_insurance_bot/app.py` (lines 76–77, 370–431)
    - **Bug:** Custom user (`ss.user_id = "self"`) is not reliably restored into context on reruns when `Demo Profile` vs `Custom Input` toggles or when switching views.

16. **False `"✅ Policy Active & Indexed"` Status & Path Traversal Risk**
    - **Location:** `health_insurance_bot/app.py` (lines 441, 448–453)
    - **Bug:** `tmp = POLICIES_DIR / "_uploads" / uploaded.name` uses unsanitized filename; UI displays `✅ Policy Active & Indexed — ⚡ 48 Structured Rules Loaded` even when rules contain 48 API errors or came from a non-insurance PDF.

17. **"AI-Generated Slop" UI & Unfriendly End-User Experience**
    - **Location:** `health_insurance_bot/app.py` (lines 80–585)
    - **Issues:**
      - Heavy `!important` CSS hacks fighting Streamlit's layout engine, harsh neon cyan/emerald gradients, emoji overload on every label.
      - Internal developer benchmark pills (`⚡ Groq LPU (~0.7s)`, `🎯 0.0% Math Error`, `🧭 96.7% Multi-Tool Routing`) displayed as hero branding to insurance end-users.
      - No **Policy Coverage Explorer** (users cannot browse extracted rules, sub-limits, waiting periods, exclusions, or citations directly).
      - No **Visual Claim Calculator** card/form for quick interactive simulations without typing long prompts.
      - Raw `st.json(obs)` dumped in the agent trace instead of clean, readable step cards.

---

### D. Environment, Config & Evaluation Bugs

18. **Broken Virtual Environments & Missing Dependencies**
    - **Location:** `.venv/`, `health_insurance_bot/.venv/`, `requirements.txt`, `health_insurance_bot/requirements.txt`
    - **Bug:** Existing `.venv`s have broken symlinks/paths referencing `/Users/dibyajyoti/...` and unsigned `pydantic_core` Python 3.14 `.so` binaries blocked by macOS policy. `requirements.txt` omits `langchain-classic` (imported in `agent.py` line 6) and `langchain-text-splitters` (imported in `ingest.py` line 17).

19. **Insecure Streamlit Server Defaults**
    - **Location:** `.streamlit/config.toml` (lines 11–12)
    - **Bug:** Sets `enableCORS = false` and `enableXsrfProtection = false`.

20. **Benchmark & Evaluation Suite Does Not Actually Test Real Agent/Tools**
    - **Location:** `scripts/evaluate_and_benchmark.py`
    - **Bug:**
      - Benchmark 1 (`line 277`) hardcodes `tool_payable = payable_ground_truth` instead of calling `calculate_claim`.
      - Benchmark 2 calls raw Groq chat completions instead of the real LangChain `AgentExecutor` ReAct loop.
      - Benchmark 3 & 4 use synthetic string prompts and `dict.get()` instead of real RAG chunks and tool executions.
      - Zero tests exist for out-of-domain PDF rejection, extraction quality, missing/unverified conditions, fuzzy-matching edge cases, or multi-turn memory.

---

## Part 2: Step-by-Step Fix & Verification Plan

### Phase 1: Clean Environment & Configuration
- [x] **Task 1.1:** Create a clean, working Python virtual environment (Python 3.12 via Homebrew) and remove stale/broken `.venv` issues.
- [x] **Task 1.2:** Update `requirements.txt` & `health_insurance_bot/requirements.txt` with exact, verified packages (`langchain-classic`, `langchain-text-splitters`, `langchain-groq`, `chromadb`, `sentence-transformers`, `pdfplumber`, `streamlit`, `pandas`, `python-dotenv`, `pytest`).
- [x] **Task 1.3:** Clean up `.streamlit/config.toml` security flags (`enableXsrfProtection`, Linear/Vercel minimalist dark theme alignment) and fix API key resolution across env / `.env` / `st.secrets`.

### Phase 2: Robust PDF Validation, Vector Search & Extraction Pipeline
- [x] **Task 2.1:** Fix `ingest.py`:
  - Switched ChromaDB collections to explicit **cosine similarity** (`metadata={"hnsw:space": "cosine"}`) with automatic legacy migration, hybrid keyword + semantic retrieval, and bounded `[0, 1]` similarity scores.
  - Guarded against empty/scanned image-only PDFs (`EmptyPDFError`).
  - Safely clamped `n_results = min(k, coll.count())` and added similarity threshold filtering.
- [x] **Task 2.2:** Added **Insurance Policy Domain Gatekeeper** (`health_insurance_bot/policy_validator.py`):
  - Two-layer validation (domain keyword heuristic + Groq semantic classifier) rejecting non-insurance PDFs (`NonPolicyDocumentError`) before indexing or rule extraction.
- [x] **Task 2.3:** Overhauled `rules_extractor.py`:
  - Replaced missing `covered=True` defaults and exception fallbacks with honest 4-state schema (`covered`, `excluded`, `not_mentioned`, `extraction_failed`).
  - Enforced page/quote verification against retrieved excerpts.
  - Switched primary extraction to `qwen/qwen3.8-27b` (via dynamic model resolution) with rolling 8k TPM rate-limit pacing and batch grouping.
  - Added v2 schema cache integrity validation (`_is_cached_rules_valid`).
- [x] **Task 2.4:** Purged poisoned caches & old L2 `chroma_store/`, cleanly re-extracted `data/policy.pdf` (48 conditions, 0 failed) and verified `Brochure_Super Star.pdf` honesty.

### Phase 3: Grounded Agent, Tools & Session Architecture
- [x] **Task 3.1:** Made `tools/_context.py` session-safe using `contextvars.ContextVar` (`bind_context`) so concurrent Streamlit tabs never overwrite each other.
- [x] **Task 3.2:** Fixed `tools/coverage.py`:
  - Replaced generic token overlap in `_find_condition()` with stopword-filtered + alias-aware matching.
  - Refuses to claim coverage or calculate payouts on `not_mentioned` / unverified conditions.
  - Added waiting-period eligibility checks (`policy_start` vs today + PED rules) in `compute_claim_payout()` & `calculate_claim`.
- [x] **Task 3.3:** Fixed `tools/policy_search.py` to compute cosine similarity (`1 - distance`) and return `no_relevant_match` below threshold.
- [x] **Task 3.4:** Updated `agent.py`: enabled real `chat_history` memory, dynamic model resolution, strict zero-page-0 citation rules, and sample-directory transparency across `hospitals.py` & `documents.py`.

### Phase 4: Professional, High-Trust Insurtech UI/UX Redesign (`app.py`)
- [x] **Task 4.1:** Redesigned UI into a minimalist Linear/Vercel-inspired dark interface (`#0F1117` / `#161922`, muted `#5E6AD2` indigo accent, clean tabular typography, zero neon slop or benchmark badge clutter).
- [x] **Task 4.2:** Fixed Sample vs Upload override bug, Custom Profile persistence across reruns, upload filename sanitization, and honest policy rule counts / rejection messages.
- [x] **Task 4.3:** Built 4 clean workspaces: **Policy Assistant** (with human-readable trace summaries + multi-turn chat), **Coverage Explorer**, **Claim Estimator**, and **Hospitals & Checklists (Sample Reference)**.

### Phase 5: Automated Testing Suite & Honest Evaluation
- [x] **Task 5.1:** Built comprehensive `pytest` suite (`tests/test_suite.py` + `tests/test_app_ui.py` using `streamlit.testing.v1.AppTest`): **13/13 passed**. *(Note: Live Chrome visual inspection via `chrome_devtools` could not attach because Chrome remote debugging wasn't enabled locally; verified UI rendering and multi-widget flows via `AppTest` + clean Streamlit server boot on `:8502`.)*
- [x] **Task 5.2:** Rewrote `scripts/evaluate_and_benchmark.py` with unmocked end-to-end execution: **15/15 (100%)** claim math accuracy, **18/18 (100%)** real `AgentExecutor` routing & zero page-0 citation hallucinations, **63.7% prompt token reduction**, **P50: 1.85s / P90: 5.79s** full ReAct latency.
