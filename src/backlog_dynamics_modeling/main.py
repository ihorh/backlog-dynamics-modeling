import streamlit as st

from backlog_dynamics_modeling.initial_data import read_sprints_data
from backlog_dynamics_modeling.project import Project

st.title("Modeling Backlog Dynamics:")
st.header("From Deterministic Trends to Probabilistic Forecasts")

project = Project(name="Cat Memes Designer", backlog_initial_size=256)

st.write(project)

df = read_sprints_data()
st.table(df)

