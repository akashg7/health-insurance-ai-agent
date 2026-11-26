"""Streamlit workspace for the Health Insurance Policy Assistant.

Designed with a minimalist Linear/Vercel-inspired dark interface:
- Four dedicated workspaces: Assistant Chat, Coverage Explorer, Claim Estimator, Reference Directory
- Strict PDF validation before indexing (rejects non-insurance or scanned PDFs safely)
- Session-isolated PolicyContext + multi-turn conversational history
- Live waiting-period-aware deterministic claim breakdown
"""

import html
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from agent import build_agent
from ingest import EmptyPDFError, pdf_fingerprint
from policy_processor import process
from policy_validator import NonPolicyDocumentError, get_groq_api_key
from tools._context import (
    PolicyContext,
    bind_context,
    load_policy_into_context,
)
from tools.coverage import compute_claim_payout

ROOT = Path(__file__).parent
POLICIES_DIR = ROOT / "policies"
DATA_DIR = ROOT / "data"
SAMPLE_PDF = DATA_DIR / "policy.pdf"
HOSPITALS_FILE = DATA_DIR / "hospitals.csv"
DOCS_FILE = DATA_DIR / "documents_rules.json"
USERS_FILE = DATA_DIR / "users.json"

st.set_page_config(
    page_title="PolicyLens — Health Insurance Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_users() -> dict:
    return json.loads(USERS_FILE.read_text())


@st.cache_data
def load_hospitals() -> pd.DataFrame:
    df = pd.read_csv(HOSPITALS_FILE)
    return df[df["cashless"] == True].drop(columns=["cashless"])  # noqa: E712


@st.cache_data
def load_doc_checklists() -> dict:
    return json.loads(DOCS_FILE.read_text())


def _esc(val: object) -> str:
    return html.escape(str(val if val is not None else ""))


def _format_wait(days: int | None) -> str:
    if days is None:
        return "Not specified"
    if days == 0:
        return "None (Day 1)"
    if days % 365 == 0:
        yrs = days // 365
        return f"{yrs} yr ({days}d)"
    if days >= 30 and days % 30 == 0:
        months = days // 30
        return f"{months} mo ({days}d)"
    return f"{days} days"


def _init_session_state():
    ss = st.session_state
    ss.setdefault("chat_history", [])
    ss.setdefault("policy_ctx", PolicyContext())
    ss.setdefault("policy_source_mode", "Sample: HDFC Optima Secure")
    ss.setdefault("processed_pdf_hash", None)
    ss.setdefault("processed_pdf_path", None)
    ss.setdefault("last_uploaded_hash", None)
    ss.setdefault("upload_error", None)
    ss.setdefault("upload_notice", None)
    ss.setdefault("profile_mode", "Demo profile")
    ss.setdefault("user_id", "U1001")
    ss.setdefault("custom_user_data", {
        "user_id": "self",
        "name": "",
        "city": "Bangalore",
        "sum_insured": 500_000,
        "co_pay": 0.10,
        "no_claim_bonus": 0.0,
        "policy_start": "2023-01-01",
        "policy_type": "Individual",
        "pre_existing": [],
    })
    ss.setdefault("agent", None)
    ss.setdefault("show_reasoning", True)
    ss.setdefault("pending_prompt", None)


_init_session_state()
ss = st.session_state


def _sync_user_to_ctx():
    ctx: PolicyContext = ss.policy_ctx
    if ss.profile_mode == "Custom details":
        profile = ss.custom_user_data
        if profile and profile.get("name", "").strip():
            ctx.custom_user = profile
            ctx.user_id = "self"
            return
    ctx.custom_user = None
    ctx.user_id = ss.user_id


def _activate_policy(pdf_path: Path, is_new_upload: bool = False):
    """Validate, index, and attach a policy PDF into this session's PolicyContext."""
    ss.upload_error = None
    ss.upload_notice = None
    h = pdf_fingerprint(pdf_path)

    if h != ss.processed_pdf_hash:
        progress_bar = st.sidebar.progress(0, text="Starting validation...")
        try:
            def _on_progress(step: int, total: int, msg: str):
                pct = min(1.0, max(0.0, step / max(total, 1)))
                progress_bar.progress(pct, text=msg)

            res = process(pdf_path, progress_callback=_on_progress)
            progress_bar.empty()
        except (EmptyPDFError, NonPolicyDocumentError) as exc:
            progress_bar.empty()
            ss.upload_error = str(exc)
            if is_new_upload:
                ss.last_uploaded_hash = h
            return
        except Exception as exc:
            progress_bar.empty()
            ss.upload_error = f"Could not process '{pdf_path.name}': {exc}"
            return

        ss.processed_pdf_hash = h
        ss.processed_pdf_path = pdf_path
        if is_new_upload:
            ss.last_uploaded_hash = h
            val = res.get("validation") or {}
            plan = val.get("plan_name") or pdf_path.name
            ss.upload_notice = f"Verified and loaded **{plan}**."

    load_policy_into_context(ss.policy_ctx, ss.processed_pdf_hash, ss.processed_pdf_path)
    if ss.agent is None and get_groq_api_key():
        ss.agent = build_agent(verbose=False)


# Minimalist Linear / Vercel Dark CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: #0F1117;
        color: #E6E8EC;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background-color: #13161F;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Typography refinement */
    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
        color: #F3F4F6;
    }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding-bottom: 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .app-title {
        font-size: 22px;
        font-weight: 600;
        color: #F9FAFB;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .app-subtitle {
        font-size: 13px;
        color: #9CA3AF;
        margin: 4px 0 0 0;
    }
    .doc-pill {
        font-size: 12px;
        font-weight: 500;
        color: #C7D2FE;
        background: rgba(94, 106, 210, 0.14);
        border: 1px solid rgba(94, 106, 210, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        white-space: nowrap;
    }

    /* Compact Metric Strip */
    .context-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 18px;
    }
    .context-cell {
        background: #161922;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .context-label {
        font-size: 11px;
        font-weight: 500;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 3px;
    }
    .context-value {
        font-size: 14px;
        font-weight: 600;
        color: #F3F4F6;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Subtle Surface Cards */
    .surface-card {
        background: #161922;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .sidebar-meta-card {
        background: #181C26;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 12.5px;
        line-height: 1.55;
        color: #D1D5DB;
        margin-top: 8px;
    }
    .sidebar-meta-card b {
        color: #F9FAFB;
        font-variant-numeric: tabular-nums;
    }

    /* Buttons (Clean Minimalist) */
    div.stButton > button {
        background-color: #1B1F2A;
        color: #E5E7EB;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 7px;
        font-size: 13px;
        font-weight: 500;
        padding: 7px 14px;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    div.stButton > button:hover {
        background-color: #242938;
        border-color: #5E6AD2;
        color: #FFFFFF;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-covered {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-excluded {
        background: rgba(239, 68, 68, 0.14);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-unknown {
        background: rgba(156, 163, 175, 0.12);
        color: #9CA3AF;
        border: 1px solid rgba(156, 163, 175, 0.25);
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background-color: #161922;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {
        font-size: 14px;
        line-height: 1.65;
        color: #E5E7EB;
    }

    /* Responsive strip on narrow screens */
    @media (max-width: 900px) {
        .context-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("### PolicyLens")
    st.caption("Grounded Health Policy Assistant")
    st.divider()

    # 1. Active Document Selector
    st.markdown("**1. Policy Document**")
    source_choice = st.radio(
        "Policy Source",
        ["Sample: HDFC Optima Secure", "Upload Policy PDF"],
        key="policy_source_mode",
        label_visibility="collapsed",
    )

    if source_choice == "Sample: HDFC Optima Secure":
        if ss.processed_pdf_path != SAMPLE_PDF:
            _activate_policy(SAMPLE_PDF, is_new_upload=False)
    else:
        uploaded = st.file_uploader(
            "Upload health insurance PDF",
            type="pdf",
            help="Upload a health insurance policy wording, schedule, or brochure (selectable text PDF, max 25 MB).",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            safe_name = Path(uploaded.name).name
            upload_dir = POLICIES_DIR / "_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = upload_dir / safe_name
            raw_bytes = uploaded.getvalue()
            tmp_path.write_bytes(raw_bytes)
            uploaded_hash = pdf_fingerprint(tmp_path)

            if uploaded_hash != ss.processed_pdf_hash and uploaded_hash != ss.last_uploaded_hash:
                _activate_policy(tmp_path, is_new_upload=True)
                st.rerun()
            elif uploaded_hash == ss.processed_pdf_hash:
                load_policy_into_context(ss.policy_ctx, ss.processed_pdf_hash, tmp_path)

    if ss.upload_error:
        st.error(ss.upload_error)
    elif ss.upload_notice:
        st.success(ss.upload_notice)

    # Display Verified Policy Status
    meta = ss.policy_ctx.policy_meta or {}
    if meta:
        val = meta.get("validation") or {}
        doc_type = (val.get("document_type") or "policy").replace("_", " ").title()
        cov_n = meta.get("covered_count", 0)
        exc_n = meta.get("excluded_count", 0)
        nm_n = meta.get("not_mentioned_count", 0)
        fail_n = meta.get("failed_count", 0)
        st.markdown(
            f"""
            <div class="sidebar-meta-card">
                <div><b>{_esc(val.get('plan_name') or meta.get('pdf_filename', 'Active Policy'))}</b></div>
                <div style="color:#9CA3AF; font-size:11.5px; margin-bottom:6px;">
                    {_esc(val.get('insurer_or_issuer') or 'Verified document')} &middot; {_esc(doc_type)}
                </div>
                <div>Covered: <b>{cov_n}</b> &middot; Excluded: <b>{exc_n}</b> &middot; Unlisted: <b>{nm_n}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if fail_n > 0:
            st.warning(f"{fail_n} condition rule(s) failed extraction due to API limits. Try re-processing.")

    st.divider()

    # 2. Policyholder Profile
    st.markdown("**2. Policyholder Profile**")
    users = load_users()
    st.radio(
        "Profile Mode",
        ["Demo profile", "Custom details"],
        key="profile_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    if ss.profile_mode == "Demo profile":
        options = list(users.keys())
        labels = {uid: f"{u['name']} ({u['city']})" for uid, u in users.items()}
        selected = st.selectbox(
            "Select profile",
            options,
            index=options.index(ss.user_id) if ss.user_id in options else 0,
            format_func=lambda x: labels[x],
            label_visibility="collapsed",
        )
        ss.user_id = selected
        u_info = users[selected]
        ped_text = ", ".join(u_info.get("pre_existing") or []) or "None"
        st.markdown(
            f"""
            <div class="sidebar-meta-card">
                <div>Sum Insured: <b>₹{u_info['sum_insured']:,}</b> &middot; Co-pay: <b>{int(u_info['co_pay']*100)}%</b></div>
                <div>Started: <b>{_esc(u_info.get('policy_start', 'N/A'))}</b></div>
                <div>Pre-existing: <b>{_esc(ped_text)}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        prev = ss.custom_user_data
        cities = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"]
        with st.form("custom_profile_form", clear_on_submit=False):
            name = st.text_input("Name", value=prev.get("name", ""))
            city_idx = cities.index(prev.get("city")) if prev.get("city") in cities else 0
            city = st.selectbox("City", cities, index=city_idx)
            sum_insured = st.number_input(
                "Sum Insured (₹)",
                min_value=100_000,
                max_value=20_000_000,
                value=int(prev.get("sum_insured", 500_000)),
                step=100_000,
            )
            co_pay_pct = st.number_input(
                "Co-pay (%)",
                min_value=0,
                max_value=50,
                value=int(round(float(prev.get("co_pay", 0.10)) * 100)),
            )
            policy_start = st.text_input(
                "Policy start date (YYYY-MM-DD)",
                value=str(prev.get("policy_start") or "2023-01-01"),
            )
            ped = st.text_input(
                "Pre-existing conditions (comma-separated)",
                value=", ".join(prev.get("pre_existing", [])),
            )
            submitted = st.form_submit_button("Apply profile", use_container_width=True)

        if submitted:
            if not name.strip():
                st.warning("Please enter a policyholder name.")
            else:
                ss.custom_user_data = {
                    "user_id": "self",
                    "name": name.strip(),
                    "city": city,
                    "sum_insured": int(sum_insured),
                    "co_pay": float(co_pay_pct) / 100.0,
                    "no_claim_bonus": float(prev.get("no_claim_bonus", 0.0)),
                    "policy_start": policy_start.strip() or None,
                    "policy_type": "Custom",
                    "pre_existing": [p.strip() for p in ped.split(",") if p.strip()],
                }
                st.success(f"Applied profile: {name.strip()}")

    _sync_user_to_ctx()

    st.divider()

    # 3. Controls
    st.markdown("**3. Controls**")
    ss.show_reasoning = st.toggle("Show tool trace", value=ss.show_reasoning)
    if st.button("Clear chat history", use_container_width=True):
        ss.chat_history = []
        st.rerun()


# Ensure default sample policy is active on initial load
if ss.processed_pdf_hash is None:
    _activate_policy(SAMPLE_PDF, is_new_upload=False)
_sync_user_to_ctx()

active_ctx: PolicyContext = ss.policy_ctx
if active_ctx.custom_user is not None:
    active_u = active_ctx.custom_user
else:
    active_u = users.get(active_ctx.user_id or "U1001", next(iter(users.values())))


# ==============================================================================
# MAIN WORKSPACE HEADER & CONTEXT BAR
# ==============================================================================
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <p class="app-title">Health Insurance Policy Assistant</p>
            <p class="app-subtitle">Citation-backed coverage verification, waiting-period checks, and deterministic claim estimates.</p>
        </div>
        <span class="doc-pill">{_esc(active_ctx.policy_label)}</span>
    </div>
    <div class="context-strip">
        <div class="context-cell">
            <div class="context-label">Policyholder</div>
            <div class="context-value">{_esc(active_u.get('name', 'Policyholder'))} ({_esc(active_u.get('city', ''))})</div>
        </div>
        <div class="context-cell">
            <div class="context-label">Sum Insured</div>
            <div class="context-value">₹{int(active_u.get('sum_insured', 0)):,}</div>
        </div>
        <div class="context-cell">
            <div class="context-label">Policy Co-pay</div>
            <div class="context-value">{int(round(float(active_u.get('co_pay', 0)) * 100))}%</div>
        </div>
        <div class="context-cell">
            <div class="context-label">Policy Start</div>
            <div class="context-value">{_esc(active_u.get('policy_start') or 'Not set')}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _to_langchain_history(history: list[dict]) -> list:
    msgs = []
    for turn in history[-10:]:  # Keep last 5 exchanges for tight prompt size
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            msgs.append(AIMessage(content=turn["content"]))
    return msgs


def _summarize_tool_step(tool_name: str, obs: object) -> str:
    if isinstance(obs, dict):
        if "error" in obs:
            return f"Error: {obs['error']}"
        if tool_name == "check_coverage":
            status = obs.get("status", "unknown")
            pg = obs.get("page_reference")
            pg_str = f" (Page {pg})" if pg else ""
            return f"{obs.get('matched_rule_name', obs.get('condition'))}: {status}{pg_str}"
        if tool_name == "calculate_claim":
            if obs.get("covered") is False:
                return "Excluded (Payable: ₹0)"
            if obs.get("eligible_today") is False:
                return f"Waiting period active ({obs.get('waiting_days_remaining')}d left) — Payable today: ₹0"
            return f"Covered ₹{obs.get('covered_amount', 0):,.0f} | Payable ₹{obs.get('payable_to_user', 0):,.0f}"
        if tool_name == "get_user_policy":
            return f"{obs.get('name', 'User')} | Sum Insured ₹{obs.get('sum_insured', 0):,} | Co-pay {int(float(obs.get('co_pay', 0))*100)}%"
        if tool_name == "find_network_hospitals":
            return f"Found {obs.get('count', 0)} sample cashless hospital(s) in {obs.get('city', '')}"
        if tool_name == "list_required_documents":
            return f"Loaded {len(obs.get('documents', []))} standard checklist items ({obs.get('claim_type', '')})"
    if isinstance(obs, list) and obs:
        first = obs[0]
        if isinstance(first, dict) and first.get("status") == "no_relevant_match":
            return "No matching clauses found above similarity threshold."
        return f"Retrieved {len(obs)} relevant policy chunk(s)"
    return "Completed"


def render_steps(steps: list):
    if not steps:
        return
    with st.expander(f"Tool execution trace ({len(steps)} step{'s' if len(steps) != 1 else ''})", expanded=False):
        for idx, (action, obs) in enumerate(steps, 1):
            summary = _summarize_tool_step(action.tool, obs)
            st.markdown(f"**{idx}. `{action.tool}`** — {summary}")
            st.caption(f"Arguments: `{json.dumps(action.tool_input)}`")
            with st.popover("View raw JSON output"):
                st.json(obs)


# ==============================================================================
# WORKSPACE TABS
# ==============================================================================
tab_chat, tab_coverage, tab_calc, tab_ref = st.tabs([
    "Policy Assistant",
    "Coverage Explorer",
    "Claim Estimator",
    "Hospitals & Checklists",
])


# --- TAB 1: CHAT ASSISTANT ---
with tab_chat:
    if not get_groq_api_key():
        st.warning("`GROQ_API_KEY` is missing. Add it to `.env` or Streamlit Secrets to chat with the assistant.")

    if not ss.chat_history:
        st.markdown("<p style='font-size:13px; color:#9CA3AF; margin-bottom:8px;'>Suggested questions for this policy:</p>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Cataract surgery & wait period", use_container_width=True):
                ss.pending_prompt = "Is cataract surgery covered under my policy, and what is the waiting period?"
        with c2:
            if st.button("Estimate ₹85k claim payout", use_container_width=True):
                ss.pending_prompt = "My cataract surgery bill is Rs 85,000. How much will I get reimbursed?"
        with c3:
            if st.button("Maternity & exclusions check", use_container_width=True):
                ss.pending_prompt = "Are maternity expenses or cosmetic procedures covered under this policy?"
        with c4:
            if st.button("Cashless hospitals & docs", use_container_width=True):
                ss.pending_prompt = "Find cashless hospitals in Bangalore for ophthalmology and what documents are needed."

    for turn in ss.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if ss.show_reasoning and turn.get("steps"):
                render_steps(turn["steps"])

    user_msg = st.chat_input("Ask about coverage, waiting periods, claim estimates, or policy clauses...")
    if ss.pending_prompt:
        user_msg = ss.pending_prompt
        ss.pending_prompt = None

    if user_msg:
        lc_history = _to_langchain_history(ss.chat_history)
        ss.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Checking policy rules..."):
                try:
                    if ss.agent is None:
                        ss.agent = build_agent(verbose=False)
                    with bind_context(ss.policy_ctx):
                        result = ss.agent.invoke({
                            "input": user_msg,
                            "chat_history": lc_history,
                            "policy_context": active_ctx.policy_label,
                        })
                    answer = result["output"]
                    steps = result.get("intermediate_steps", [])
                except Exception as exc:
                    answer = f"**Unable to process request:** `{exc}`\n\nVerify `GROQ_API_KEY` and API rate limits."
                    steps = []

            st.markdown(answer)
            if ss.show_reasoning and steps:
                render_steps(steps)

        ss.chat_history.append({"role": "assistant", "content": answer, "steps": steps})
        st.rerun()


# --- TAB 2: COVERAGE EXPLORER ---
with tab_coverage:
    rules = active_ctx.rules or {}
    if not rules:
        st.info("No structured coverage rules are available for this document.")
    else:
        col_f1, col_f2 = st.columns([2, 3])
        with col_f1:
            status_filter = st.selectbox(
                "Filter by coverage status",
                ["All", "Covered", "Excluded", "Not Mentioned"],
                index=0,
            )
        with col_f2:
            query_filter = st.text_input("Search condition or procedure", placeholder="e.g. cataract, maternity, ICU, dental...")

        rows = []
        for cond_name, r in rules.items():
            st_val = r.get("status") or ("covered" if r.get("covered") else ("excluded" if r.get("covered") is False else "not_mentioned"))
            label_status = {"covered": "Covered", "excluded": "Excluded", "not_mentioned": "Not Mentioned"}.get(st_val, "Error")

            if status_filter != "All" and label_status != status_filter:
                continue
            if query_filter.strip() and query_filter.strip().lower() not in cond_name.lower() and query_filter.strip().lower() not in (r.get("notes") or "").lower():
                continue

            sub_val = r.get("sub_limit_inr")
            rows.append({
                "Condition / Benefit": cond_name,
                "Status": label_status,
                "Waiting Period": _format_wait(r.get("waiting_period_days")),
                "Sub-Limit": f"₹{int(sub_val):,}" if sub_val else "None / At actuals",
                "Page": r.get("page_reference") or "—",
                "Notes": r.get("notes") or "",
                "Quote": r.get("evidence_quote") or "",
            })

        st.caption(f"Showing {len(rows)} of {len(rules)} audited policy conditions for **{active_ctx.policy_label}**.")
        if rows:
            df_rules = pd.DataFrame(rows)
            st.dataframe(
                df_rules[["Condition / Benefit", "Status", "Waiting Period", "Sub-Limit", "Page", "Notes"]],
                use_container_width=True,
                hide_index=True,
                height=440,
            )
            with st.expander("Inspect exact policy evidence quotes", expanded=False):
                for row in rows:
                    if row["Quote"]:
                        st.markdown(f"- **{row['Condition / Benefit']}** *(Page {row['Page']})*: `\"{row['Quote']}\"`")
        else:
            st.info("No conditions matched your current filter.")


# --- TAB 3: DETERMINISTIC CLAIM ESTIMATOR ---
with tab_calc:
    st.markdown("Estimate reimbursement using exact policy rules and your active policyholder profile.")
    rules = active_ctx.rules or {}
    verified_conditions = [
        k for k, v in rules.items()
        if v.get("status") in {"covered", "excluded"} or v.get("covered") is not None
    ]
    if not verified_conditions:
        st.info("No verified conditions found in the active document to run a claim estimate.")
    else:
        c_left, c_right = st.columns([1, 1], gap="large")
        with c_left:
            selected_cond = st.selectbox("Medical condition or procedure", verified_conditions)
            bill_inr = st.number_input("Estimated Hospital Bill (₹)", min_value=1000, max_value=50_000_000, value=75000, step=5000)
            rule_obj = rules[selected_cond]
            st.markdown(
                f"""
                <div class="surface-card">
                    <div style="font-size:12px; color:#9CA3AF; margin-bottom:4px;">EXTRACTED POLICY RULE</div>
                    <div><b>{_esc(selected_cond)}</b> &middot; Page {_esc(rule_obj.get('page_reference') or '—')}</div>
                    <div style="font-size:13px; color:#D1D5DB; margin-top:6px;">{_esc(rule_obj.get('notes', ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_right:
            payout = compute_claim_payout(rule_obj, active_u, float(bill_inr), selected_cond, selected_cond)
            if "error" in payout:
                st.warning(payout["error"])
            elif not payout.get("covered"):
                st.markdown(
                    f"""
                    <div class="surface-card">
                        <span class="badge badge-excluded">EXCLUDED</span>
                        <h3 style="margin:10px 0 4px 0;">Payable Today: ₹0</h3>
                        <p style="color:#9CA3AF; font-size:13px; margin:0;">{_esc(payout.get('reason'))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                wait_status = payout.get("waiting_period_status")
                elig = payout.get("eligible_today")
                if elig is False:
                    st.warning(
                        f"Waiting period active ({payout['waiting_period_days']} days required; "
                        f"{payout['waiting_days_remaining']} days remaining based on start date {active_u.get('policy_start')}). "
                        f"Payable today is ₹0. Estimated payout once cleared: ₹{payout['estimated_payout_after_wait']:,.0f}."
                    )
                elif wait_status == "verify_policy_start_date":
                    st.info(
                        f"This condition carries a {payout['waiting_period_days']}-day waiting period. "
                        "Verify your policy start date to confirm current eligibility."
                    )

                st.markdown(
                    f"""
                    <div class="surface-card">
                        <span class="badge badge-covered">COVERED</span>
                        <div style="margin-top:12px; display:grid; grid-template-columns:1fr 1fr; gap:10px; font-variant-numeric:tabular-nums;">
                            <div><span style="color:#9CA3AF; font-size:12px;">Hospital Bill</span><br><b>₹{payout['bill_amount']:,.0f}</b></div>
                            <div><span style="color:#9CA3AF; font-size:12px;">Applied Cap</span><br><b>{_esc(payout['sub_limit_applied'])}</b></div>
                            <div><span style="color:#9CA3AF; font-size:12px;">Eligible Covered Amount</span><br><b>₹{payout['covered_amount']:,.0f}</b></div>
                            <div><span style="color:#9CA3AF; font-size:12px;">Co-pay ({payout['co_pay_percent']:.0f}%)</span><br><b>- ₹{payout['co_pay_amount']:,.0f}</b></div>
                        </div>
                        <hr style="border-color:rgba(255,255,255,0.08); margin:12px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:baseline; font-variant-numeric:tabular-nums;">
                            <span style="color:#9CA3AF; font-size:13px;">Payable Today</span>
                            <span style="font-size:22px; font-weight:700; color:#F9FAFB;">₹{payout['payable_to_user']:,.0f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# --- TAB 4: REFERENCE HOSPITALS & CHECKLISTS ---
with tab_ref:
    st.info(
        "**Sample Reference Data:** Hospital network entries and document checklists below come from the "
        "built-in sample directory and IRDAI standard guidelines (not extracted from your uploaded PDF)."
    )
    ref_left, ref_right = st.columns([3, 2], gap="large")
    with ref_left:
        st.markdown("#### Cashless Hospital Directory (Sample)")
        hosp_df = load_hospitals()
        h1, h2 = st.columns(2)
        with h1:
            sel_city = st.selectbox("City", ["All"] + sorted(hosp_df["city"].unique().tolist()))
        with h2:
            sel_spec = st.text_input("Specialty filter", placeholder="e.g. ophthalmology, cardiology")

        filtered = hosp_df.copy()
        if sel_city != "All":
            filtered = filtered[filtered["city"] == sel_city]
        if sel_spec.strip():
            filtered = filtered[filtered["specialties"].str.lower().str.contains(sel_spec.strip().lower(), na=False)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=340)

    with ref_right:
        st.markdown("#### Standard Claim Document Checklists")
        docs_map = load_doc_checklists()
        claim_choice = st.selectbox(
            "Claim category",
            list(docs_map.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        for item in docs_map[claim_choice]:
            st.markdown(f"- {item}")
