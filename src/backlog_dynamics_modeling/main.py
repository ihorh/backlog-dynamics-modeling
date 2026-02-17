from typing import Final

import streamlit as st

from backlog_dynamics_modeling.config import GITHUB_REPO_URL

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

with st.sidebar.container(key="sidebar_bottom"):
    st.divider()
    st.caption(f"[View the source on GitHub]({GITHUB_REPO_URL})")
    st.caption("&copy; 2026 Ihor H.")

st.html("""
<style>
    .st-key-sidebar_bottom {
        position: absolute;
        bottom: 10px;
    }
</style>
""")

st.divider()

st.caption("""
This content is provided for educational purposes only and should not be considered financial or legal advice.
""")
