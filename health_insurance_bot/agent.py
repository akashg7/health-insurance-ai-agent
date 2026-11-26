"""LangChain tool-calling ReAct agent factory with multi-turn memory and strict grounding."""

import os

from dotenv import load_dotenv
from groq import Groq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from policy_validator import get_groq_api_key, resolve_groq_model
from tools.coverage import calculate_claim, check_coverage
from tools.documents import list_required_documents
from tools.hospitals import find_network_hospitals
from tools.policy_search import search_policy_docs
from tools.user import get_user_policy

load_dotenv()

AGENT_MODEL_PREFERENCE = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

SYSTEM_PROMPT = """You are a grounded Health Insurance Policy Assistant helping a policyholder understand their active uploaded PDF.

ACTIVE POLICY CONTEXT:
{policy_context}

TOOL DECISION POLICY (6 tools):
1. Specific condition/surgery coverage or waiting period ("Is X covered?") -> call `check_coverage` first.
   - IMPORTANT: If `check_coverage` returns `status="not_mentioned"` or `covered=null` (or an error), you MUST immediately call `search_policy_docs` to inspect the full text of the active PDF before answering!
2. Open-ended "why/how/explain/define" questions or clauses -> call `search_policy_docs`.
3. Claim payout or reimbursement estimate ("How much will I get for ₹X?") -> chain: `check_coverage` -> `get_user_policy` -> `calculate_claim`.
   - If `check_coverage` shows `covered=null`, explain honestly that you cannot calculate a verified claim without explicit coverage confirmation in the PDF.
   - If `calculate_claim` reports `eligible_today=false` because the waiting period has not completed yet (`waiting_period_not_completed`), lead clearly with **₹0 payable today (waiting period active: X days remaining)** and also share the post-waiting-period payout estimate!
4. Hospital lookup ("Where can I go?" / "cashless hospitals") -> call `find_network_hospitals`.
   - Always clarify concisely that hospital directory results come from the built-in **sample cashless reference directory** (not extracted from the uploaded PDF).
5. Required paperwork ("What documents do I need?") -> call `list_required_documents` (note that this reflects the standard IRDAI reference checklist).
6. User profile, sum insured, co-pay, or policy start date -> call `get_user_policy`.

STRICT GROUNDEDNESS & CITATION RULES:
- NEVER invent, guess, or extrapolate coverage, sub-limits, co-pays, waiting periods, or page numbers.
- If BOTH `check_coverage` and `search_policy_docs` find no mention of the topic in the active PDF (e.g. `no_relevant_match`), state clearly and honestly: *"This document does not mention coverage or terms for <topic>."*
- NEVER output "Source: policy page 0" or "page None". Only cite a page number when a tool explicitly returned a positive integer `page_reference` / `page`. If the answer comes from `hospitals` or `documents` reference data, cite that instead of a page number.
- Format all currency in INR with commas (e.g., `₹36,000`).

RESPONSE STYLE:
- Lead with a clear, direct 1–2 sentence takeaway.
- Use crisp bullet points or a compact key-value breakdown when presenting claim math, waiting periods, documents, or hospital lists.
- End with a short grounded source line when applicable (e.g., `*Source: Policy Page 29*` or `*Source: Built-in Sample Directory*`)."""


def build_agent(verbose: bool = False) -> AgentExecutor:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment / .env / Streamlit secrets")

    client = Groq(api_key=api_key, timeout=30.0)
    model = resolve_groq_model(client, "GROQ_MODEL", AGENT_MODEL_PREFERENCE)
    llm = ChatGroq(model=model, temperature=0, api_key=api_key, timeout=30.0)

    tools = [
        check_coverage,
        calculate_claim,
        get_user_policy,
        find_network_hospitals,
        list_required_documents,
        search_policy_docs,
    ]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=8,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from ingest import pdf_fingerprint
    from tools._context import get_context, set_policy, set_user

    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "policy.pdf"
    user_id = sys.argv[2] if len(sys.argv) > 2 else "U1001"

    set_policy(pdf_fingerprint(pdf), pdf)
    set_user(user_id)
    ctx = get_context()

    executor = build_agent(verbose=False)
    questions = [
        "Is cataract surgery covered? My bill is Rs 50,000 — how much will I get?",
        "What documents do I need to file a hospitalization reimbursement claim?",
        "Find a cashless hospital in Bangalore for ophthalmology.",
    ]
    for q in questions:
        print(f"\n>>> USER: {q}")
        result = executor.invoke({"input": q, "chat_history": [], "policy_context": ctx.policy_label})
        print(f"AGENT: {result['output']}")
        steps = result.get("intermediate_steps", [])
        print(f"  ({len(steps)} tool call(s): {[s[0].tool for s in steps]})")
