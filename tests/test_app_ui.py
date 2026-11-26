"""End-to-end Streamlit AppTest UI & state regression tests."""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_FILE = ROOT_DIR / "health_insurance_bot" / "app.py"


def test_app_boots_cleanly_with_verified_policy_and_tabs():
    at = AppTest.from_file(str(APP_FILE), default_timeout=20)
    at.run()

    assert not at.exception, f"App raised exception on boot: {at.exception}"
    assert at.session_state["processed_pdf_hash"] == "79662228bf4a3046"
    assert len(at.tabs) == 4
    # Sidebar metadata should show verified rules breakdown without errors
    sidebar_md = " ".join(m.value for m in at.sidebar.markdown)
    assert "PolicyLens" in sidebar_md
    assert "Covered:" in sidebar_md
    assert "Excluded:" in sidebar_md


def test_profile_switching_and_custom_profile_persistence():
    at = AppTest.from_file(str(APP_FILE), default_timeout=20)
    at.run()
    assert not at.exception

    # Switch to Custom details via sidebar radio
    profile_radio = at.sidebar.radio[1]
    profile_radio.set_value("Custom details").run()
    assert not at.exception
    assert at.session_state["profile_mode"] == "Custom details"

    # Fill custom form inputs
    at.sidebar.text_input[0].set_value("Akash Test")
    at.sidebar.number_input[0].set_value(1_200_000)
    at.sidebar.number_input[1].set_value(15)
    at.sidebar.text_input[1].set_value("2022-05-01")
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.session_state["policy_ctx"].user_id == "self"
    assert at.session_state["policy_ctx"].custom_user["name"] == "Akash Test"
    assert at.session_state["policy_ctx"].custom_user["sum_insured"] == 1_200_000


def test_coverage_explorer_and_claim_estimator_widgets():
    at = AppTest.from_file(str(APP_FILE), default_timeout=20)
    at.run()
    assert not at.exception

    # Filter Coverage Explorer tab to Excluded items
    at.tabs[1].selectbox[0].set_value("Excluded").run()
    assert not at.exception

    # Select cataract surgery in Claim Estimator tab and compute Rs 1,00,000
    at.tabs[2].selectbox[0].set_value("cataract surgery")
    at.tabs[2].number_input[0].set_value(100_000).run()
    assert not at.exception

    calc_md = " ".join(m.value for m in at.tabs[2].markdown)
    assert "Payable Today" in calc_md
    assert "90,000" in calc_md  # Asha Sharma 10% copay on 100,000 -> 90,000


def test_policy_source_switching_between_sample_and_upload():
    at = AppTest.from_file(str(APP_FILE), default_timeout=20)
    at.run()
    assert not at.exception

    # Switch to Upload Policy PDF mode cleanly
    at.sidebar.radio[0].set_value("Upload Policy PDF").run()
    assert not at.exception
    assert at.session_state["policy_source_mode"] == "Upload Policy PDF"

    # Switch back to Sample mode cleanly without widget conflict or state loss
    at.sidebar.radio[0].set_value("Sample: HDFC Optima Secure").run()
    assert not at.exception
    assert at.session_state["policy_source_mode"] == "Sample: HDFC Optima Secure"
    assert at.session_state["processed_pdf_hash"] == "79662228bf4a3046"
