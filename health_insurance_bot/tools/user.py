"""User policy profile lookup tool."""

import json
from langchain_core.tools import tool

from ._context import DATA_DIR, get_context

USERS_FILE = DATA_DIR / "users.json"


def _get_user_data(user_id: str | None) -> dict | None:
    ctx = get_context()
    # Custom user takes precedence only when no explicit demo user_id override was requested
    if (not user_id or user_id == "self") and ctx.custom_user is not None:
        return {**ctx.custom_user, "_profile_source": "custom_user_input"}
    uid = user_id or ctx.user_id
    if not uid:
        return None
    users = json.loads(USERS_FILE.read_text())
    found = users.get(uid)
    if found is None and uid == "self" and ctx.custom_user is not None:
        return {**ctx.custom_user, "_profile_source": "custom_user_input"}
    if found is not None:
        return {**found, "_profile_source": "sample_demo_profile"}
    return None


@tool
def get_user_policy(user_id: str = "") -> dict:
    """Get the active policyholder's details: sum insured, co-pay percentage, no-claim bonus,
    policy start date, city, and declared pre-existing conditions.

    Use this whenever you need user-specific values for claim calculation or waiting-period
    eligibility checks, or when the user asks about their own policy limits/profile.
    Leave user_id empty to read the active user in the current session.
    """
    ctx = get_context()
    uid = user_id.strip() or (ctx.user_id or "")
    user = _get_user_data(uid)
    if user is None:
        return {"error": f"No policyholder profile found for id '{uid}'."}
    return {"user_id": uid, **user}
