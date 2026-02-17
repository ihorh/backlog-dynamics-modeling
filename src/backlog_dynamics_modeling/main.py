from typing import Final

import streamlit as st

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
