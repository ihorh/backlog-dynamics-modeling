import streamlit as st

pg = st.navigation(
    [
        st.Page(
            "pages/page_00_intro.py",
            title="0. Introduction",
            default=True,
            url_path="/main",
        ),
    ],
)

pg.run()
