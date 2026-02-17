import streamlit as st

pg = st.navigation(
    [
        # * current version
        st.Page(
            "pages/page_00_intro.py",
            title="0. Introduction",
            default=True,
            url_path="/main",
        ),
        # * v1
        # st.Page("pages_v1/page_00_intro.py", title="0. Introduction"),
        # st.Page("pages_v1/page_01_the_question.py", title="1. The Question"),
        # st.Page("pages_v1/page_02_disclaimers.py", title="2. Disclaimers"),
        # st.Page("pages_v1/page_03_two_random_walks.py", title="3. Two Random Walks"),
    ],
)

pg.run()
