"""Session-safe context management using contextvars for concurrent Streamlit sessions + CLI."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
POLICIES_DIR = Path(__file__).parent.parent / "policies"


@dataclass
class PolicyContext:
    pdf_path: Path | None = None
    pdf_hash: str | None = None
    rules: dict = field(default_factory=dict)
    policy_meta: dict = field(default_factory=dict)
    user_id: str | None = None
    custom_user: dict | None = None

    @property
    def policy_label(self) -> str:
        val = self.policy_meta.get("validation") or {}
        insurer = val.get("insurer_or_issuer")
        plan = val.get("plan_name")
        filename = self.policy_meta.get("pdf_filename") or (self.pdf_path.name if self.pdf_path else "No PDF")
        if insurer and plan:
            return f"{insurer} — {plan} ({filename})"
        if plan:
            return f"{plan} ({filename})"
        return str(filename)


_DEFAULT_CTX = PolicyContext()
_ACTIVE_CTX: ContextVar[PolicyContext | None] = ContextVar("active_policy_ctx", default=None)


def get_context() -> PolicyContext:
    """Return the session-scoped context if bound, else the process-default context."""
    ctx = _ACTIVE_CTX.get()
    return ctx if ctx is not None else _DEFAULT_CTX


@contextmanager
def bind_context(ctx: PolicyContext):
    """Temporarily bind a session-specific PolicyContext for safe multi-user execution."""
    token = _ACTIVE_CTX.set(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE_CTX.reset(token)


def load_policy_into_context(ctx: PolicyContext, pdf_hash: str, pdf_path: Path | None = None) -> None:
    ctx.pdf_hash = pdf_hash
    ctx.pdf_path = pdf_path
    rules_file = POLICIES_DIR / pdf_hash / "rules.json"
    if rules_file.exists():
        data = json.loads(rules_file.read_text())
        ctx.rules = data.get("rules", {})
        ctx.policy_meta = data.get("_meta", {})
    else:
        ctx.rules = {}
        ctx.policy_meta = {}


def set_policy(pdf_hash: str, pdf_path: Path | None = None) -> None:
    load_policy_into_context(get_context(), pdf_hash, pdf_path)


def set_user(user_id: str) -> None:
    ctx = get_context()
    ctx.user_id = user_id
    ctx.custom_user = None


def set_custom_user(profile: dict) -> None:
    """Use the user's custom typed profile instead of a demo entry from users.json."""
    ctx = get_context()
    ctx.custom_user = profile
    ctx.user_id = profile.get("user_id", "self")
