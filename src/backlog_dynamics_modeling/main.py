import streamlit as st

from backlog_dynamics_modeling.project import Project

st.title("Modeling Backlog Dynamics:")
st.header("From Deterministic Trends to Probabilistic Forecasts")

project = Project(name="Cat Memes Designer", backlog_initial_size=256)

st.write(project)