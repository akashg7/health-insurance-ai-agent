"""Network hospital lookup tool over the sample cashless directory."""

import pandas as pd
from langchain_core.tools import tool

from ._context import DATA_DIR

HOSP_FILE = DATA_DIR / "hospitals.csv"


@tool
def find_network_hospitals(city: str, specialty: str = "") -> dict:
    """Find CASHLESS network hospitals in a given city, optionally filtered by specialty.

    Returns results from the built-in sample cashless hospital directory (for demonstration).
    Always returns ONLY cashless-eligible hospitals.

    Args:
        city: One of Bangalore, Mumbai, Delhi, Chennai, Hyderabad, Pune, Kolkata.
        specialty: Optional. e.g. cardiology, oncology, orthopedics, ophthalmology,
            neurology, urology, gynecology, pediatrics, gastroenterology. Empty = any.
    """
    df = pd.read_csv(HOSP_FILE)
    df = df[df["cashless"] == True]  # noqa: E712
    df = df[df["city"].str.lower() == city.strip().lower()]
    if specialty.strip():
        df = df[df["specialties"].str.lower().str.contains(specialty.strip().lower(), na=False)]

    if df.empty:
        return {
            "data_source": "Sample Cashless Hospital Directory (reference dataset, not extracted from uploaded PDF)",
            "city": city,
            "specialty": specialty or "any",
            "hospitals": [],
            "message": f"No cashless network hospitals found in {city}"
            + (f" for {specialty}" if specialty else ""),
        }

    records = df.drop(columns=["cashless"]).to_dict(orient="records")
    return {
        "data_source": "Sample Cashless Hospital Directory (reference dataset, not extracted from uploaded PDF)",
        "city": city,
        "specialty": specialty or "any",
        "count": len(records),
        "hospitals": records,
    }
