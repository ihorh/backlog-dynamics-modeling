from typing import Final

import streamlit as st

from backlog_dynamics_modeling.config import GITHUB_README_URL, GITHUB_REPO_URL

APP_TITLE: Final[str] = "Modeling Backlog Dynamics"

pg = st.navigation(
    [
        st.Page(
            "pages/page_00_intro.py",
            title="Probabilistic Forecast",
            default=True,
            url_path="/main",
        ),
        st.Page(
            "pages/page_01_inv_gauss.py",
            title="(WIP) Deriving the Inverse Gaussian Solution",
            url_path="/inv_gauss",
        ),
    ],
)

pg.run()

# * ========================================
# * Footers
# * ========================================

LICENSE_NOTE_SHORT: Final[str] = f"&copy; 2026 Ihor H. · [See README.md]({GITHUB_README_URL})"
LICENSE_NOTE_LONG: Final[str] = "&copy; 2026 Ihor H. · Code licensed under MIT · Article text licensed under CC BY 4.0"

with st.sidebar.container(key="sidebar_bottom"):
    st.divider()
    st.caption(f"[View the source on GitHub]({GITHUB_REPO_URL})")
    st.caption(LICENSE_NOTE_SHORT)

st.html("""
<style>
    .st-key-sidebar_bottom {
        position: absolute;
        bottom: 10px;
    }
</style>
""")

st.divider()

st.caption(f"""
{LICENSE_NOTE_LONG}

This content is provided for educational purposes only and should not be considered financial or legal advice.
""")
