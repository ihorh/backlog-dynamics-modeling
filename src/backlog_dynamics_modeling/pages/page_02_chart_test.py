import numpy as np
import streamlit as st

from backlog_dynamics_modeling.computed_data import run_simulations_cache_results
from backlog_dynamics_modeling.initial_data import (
    BACKLOG_INITIAL_SIZE,
    NUMBER_OF_SIMULATIONS,
    PRJ_NAME,
    read_sprints_data,
)
from backlog_dynamics_modeling.project.pdpd_chart import (
    ChartProjectDurationDistribution,
    PDProbModelPlotType,
)
from backlog_dynamics_modeling.project.pdpd_model import (
    PDProbDModelDiscrete,
    PDProbDModelInvGauss,
    PDProbDModelNormal,
)
from backlog_dynamics_modeling.project.project import Project

n_base_sprints = 5

project = Project(name=PRJ_NAME, backlog_initial_size=BACKLOG_INITIAL_SIZE)
project.add_sprints_data(read_sprints_data())
b0 = project.backlog_initial_size
vs = project.sprint_team_velocity[:n_base_sprints]
ds = project.sprint_backlog_delta[:n_base_sprints]
xs = np.array(vs) - np.array(ds)
x_mean, x_std = xs.mean(), xs.std(ddof=1)

sim_results = run_simulations_cache_results(project, base_sprints=n_base_sprints, n=NUMBER_OF_SIMULATIONS)

durations = sim_results["duration"].to_numpy()
d_min, d_max = durations.min(), durations.max()

model_sim = PDProbDModelDiscrete(label="Model", ds=durations)
model_norm = PDProbDModelNormal(label="Norm", mean=model_sim.mean, std=model_sim.ds.std())
model_ig_fit = PDProbDModelInvGauss.fit_data(durations, label="IG Fit")
model_ig_fpt = PDProbDModelInvGauss.first_passage_time(a=b0, mean=x_mean, std=x_std, label="IG FPT")

chart = ChartProjectDurationDistribution(x_min=d_min, x_max=d_max)
chart.plot_model(model_sim, color="indigo", types=PDProbModelPlotType.ALL)
chart.plot_model(model_norm, color="orange", types=PDProbModelPlotType.ALL)
chart.plot_model(model_ig_fit, color="green", types=PDProbModelPlotType.PDF | PDProbModelPlotType.CDF)
chart.plot_model(
    model_ig_fpt,
    color="red",
    types=PDProbModelPlotType.PDF | PDProbModelPlotType.CDF | PDProbModelPlotType.MEAN,
)

st.pyplot(chart.get_figure()[0])
