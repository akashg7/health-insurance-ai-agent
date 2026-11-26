# PolicyLens — Health Insurance AI Agent

A grounded, multi-tool conversational AI workspace for analyzing health insurance policies. Built with LangChain, Groq, ChromaDB, and Streamlit, pairing **two-stage RAG extraction** with **deterministic Python claim calculations** and strict document validation.

## Features

- **Document Validation Gatekeeper:** Screens uploaded PDFs before indexing—rejects unreadable/scanned image PDFs and unrelated non-insurance documents (resumes, invoices, non-health policies) rather than inventing coverage rules.
- **Policy Assistant (Multi-Turn Chat):** ReAct tool-calling agent with conversational memory across turns, human-readable tool execution traces, and strict page-grounded citations.
- **Coverage Explorer:** Interactive searchable and filterable breakdown of extracted benefits, sub-limits, waiting periods, exclusions, and evidence quotes from the active PDF.
- **Deterministic Claim Estimator:** Exact rupee payout calculations incorporating procedure sub-limits, policy co-pays, sum-insured caps, pre-existing conditions, and waiting-period eligibility checks against policy inception dates.
- **Sample Network & Claim Checklists:** Filterable demo cashless hospital directory (`hospitals.csv`) and standard IRDAI document checklists (`documents_rules.json`), clearly labeled as sample reference data.

## Tech Stack

- **LLM Inference:** [Groq](https://groq.com/) (`openai/gpt-oss-120b` for tool-calling agent, `qwen/qwen3.8-27b` for rule extraction, `openai/gpt-oss-20b` for document validation, with automatic available-model resolution)
- **Agent Orchestration:** LangChain Tool-Calling ReAct Agent (`AgentExecutor` + 6 specialized tools + session-scoped `contextvars`)
- **Vector Store & Embeddings:** ChromaDB (cosine similarity) + `sentence-transformers/all-MiniLM-L6-v2` with hybrid semantic + lexical retrieval
- **Document Parsing:** `pdfplumber` + `RecursiveCharacterTextSplitter` (500 chars, 75 overlap)
- **Frontend:** Streamlit (minimalist dark interface)

## Getting Started

### 1. Prerequisites
- Python 3.12 recommended
- A Groq API key from [console.groq.com](https://console.groq.com/keys)

### 2. Setup

```bash
git clone https://github.com/akashg7/health-insurance-ai-agent.git
cd health-insurance-ai-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` inside `health_insurance_bot/` (or project root):
```env
GROQ_API_KEY=gsk_your_api_key_here
```

### 3. Run the Application

```bash
./.venv/bin/streamlit run health_insurance_bot/app.py
```

Open `http://localhost:8501`.

### 4. Tests & Benchmarks

Run unit, RAG, and Streamlit `AppTest` UI tests:
```bash
./.venv/bin/pytest -v tests/
```

Run the unmocked end-to-end evaluation harness:
```bash
./.venv/bin/python scripts/evaluate_and_benchmark.py
```

## Documentation
- **[EVALUATION_REPORT.md](EVALUATION_REPORT.md):** Real unmocked benchmark results (100% deterministic math on 15 scenarios, 100% routing & zero page-0 hallucinations across 18 real agent queries, 63.7% token reduction).
- **[BUGS_AND_FIX_PLAN.md](BUGS_AND_FIX_PLAN.md):** Root-cause audit and resolution log.
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md):** Detailed architectural walkthrough.
