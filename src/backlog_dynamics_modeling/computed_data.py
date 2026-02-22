from functools import partial

import pandas as pd
import streamlit as st

from backlog_dynamics_modeling.project.project import Project, simulate_project


@st.cache_data
def run_simulations_cache_results(
    project: Project,
    *,
    n: int,
    base_sprints: int = 5,
    max_sprints: int = 100,
) -> pd.DataFrame:
    fn = partial(simulate_project, project=project, base_sprints=base_sprints, max_sprints=max_sprints)
    return pd.DataFrame([fn(rng_seed=i) for i in range(n)])
