from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import invgauss

from backlog_dynamics_modeling.config import GITHUB_REPO_URL
from backlog_dynamics_modeling.initial_data import BACKLOG_INITIAL_SIZE, CURRENT_SPRINT, PRJ_NAME, read_sprints_data
from backlog_dynamics_modeling.project.charts import chart_distribution_histogram, prj_chart_backlog_trajectory
from backlog_dynamics_modeling.project.project import (
    DeterministicProjectEstimator,
    Project,
    ProjectSimResult,
    simulate_project,
)

st.title("Modeling Backlog Dynamics:")
st.header("From Deterministic Trends to Probabilistic Forecasts")

# * ========================================
# * Intro
# * ========================================

r"""
> Recently, I've been exploring stochastic processes and their applications in finance.
> While simulating stock price paths, I quickly noticed how similar the underlying
> mechanics were to a project management problem I had modelled years ago in spreadsheets:
> answering the seemingly simple but persistently difficult question - “When will it be done?”
>
> I decided to recreate that earlier model in Python — and a few hours later,
> I was ready to publish the results.
"""

st.subheader("About")

st.markdown(f"💻 **[Check out the code on GitHub]({GITHUB_REPO_URL})**")

# * ========================================
# * When will it be done?
# * ========================================

st.header("When will it be done? ⏱️")
r"""
This question is asked regularly — by product managers, stakeholders, and leadership.

Here, I focus exclusively on the quantitative side of forecasting project completion,
deliberately simplifying reality to reduce the system to a tractable model.

While estimation practices, team dynamics, and process design clearly matter,
a coherent numerical forecast is essential for aligning expectations; without it,
even a well-executed project delivered in reasonable time can feel late or disappointing.
"""
with st.container(border=True):
    r"""
    💡 :blue[In project management producing a coherent numerical forecast is essential for aligning expectations.]
    """
r"""
Although the examples come from software, the formulation applies to any domain where
both workload and production capacity vary over time.
"""

# * ========================================
# * The Project
# * ========================================

st.header("The Project 🛠️")

r"""
A team begins work on a new project. As is typical, the scope is only partially defined:
some features are well specified, others are little more than short descriptions.
There is a shared understanding of the goal, but the path to reach it is not fully known in advance.

Work proceeds iteratively. The backlog is reviewed and refined, items are added or removed,
priorities shift, and estimates evolve as the team learns more. Delivery and scope change in parallel.

At the start, there is only a rough expectation of duration. After three to five sprints, however,
stakeholders usually ask for greater clarity — a more precise and defensible estimate
of when the project will be completed.
"""
with st.container(border=True):
    r"""
    🤷 :blue[Even in the early project stages, when exact dates are impossible, stakeholders expect clarity.
    We need quantitative tools which will let us give a realistic range of possible completion times.]
    """

# * ========================================
# * First Attempt
# * ========================================

st.header("First Attempt 🧮")

data = read_sprints_data()
backlog_size_5s = BACKLOG_INITIAL_SIZE - data[:CURRENT_SPRINT]["v"].sum() + data[:CURRENT_SPRINT]["d"].sum()
project = Project(name=PRJ_NAME, backlog_initial_size=BACKLOG_INITIAL_SIZE)
project.add_sprints_data(data[:CURRENT_SPRINT])
est = DeterministicProjectEstimator(project, base_sprints_number=5, max_sprints_number=36)

rf"""
At Sprint 0, the backlog consisted of `{BACKLOG_INITIAL_SIZE}` story points.
After `{CURRENT_SPRINT}` sprints, it has decreased to `{backlog_size_5s}`.

Here is data for the first {CURRENT_SPRINT} sprints:
"""

st.dataframe(
    data[:CURRENT_SPRINT][["v", "d"]]
    .rename(columns={"v": "Story Points Completed", "d": "Story Points Added / Removed"})
    .T.rename_axis("Sprint"),
)
v_mean = data[:CURRENT_SPRINT]["v"].mean()
d_mean = data[:CURRENT_SPRINT]["d"].mean()
v_std = data[:CURRENT_SPRINT]["v"].std()
d_std = data[:CURRENT_SPRINT]["d"].std()
rf"""
The average velocity over these sprints is `{v_mean:.2f}` story points per sprint.
But we need also account for changes in the backlog, which gives us a **net velocity** of
`{v_mean - d_mean:.2f}` story points per sprint.
Assuming this rate continues and the scope remains fixed, a straightforward extrapolation yields an estimated
total duration of approximately `33` sprints.

This implies a linear burn-down trajectory:
"""
fig = prj_chart_backlog_trajectory(bs=est.backlog_size())
st.pyplot(fig)

est = DeterministicProjectEstimator(project, base_sprints_number=5, max_sprints_number=8)
fig = prj_chart_backlog_trajectory(bs=est.backlog_size(), chart_subtitle="zoom-in")
st.pyplot(fig)
st.caption("""Note: sprint numbers are 0-based, so e.g. 33rd sprint has number 32.""")
r"""
However, even after just five sprints, a concern emerges: why should we assume the net velocity
will continue at exactly the same level?

The average does capture variation in a basic sense - it is neither the best nor the worst
sprint - but it compresses all possible trajectories into a single number.
"""

st.warning("""
A completion date derived from simple average cannot be treated as guaranteed.
Even if the input data perfectly reflects variability in velocity and backlog
changes, this number ignores how outcomes could spread. Relying on it alone
creates a false sense of certainty.
""")

r"""
Understanding the range of possible results requires a different approach to analyzing the data.
"""

# * ========================================
# * The Simulation
# * ========================================

st.header("The Simulation 🎲🎲")
r"""
Imagine running the project not just once, but many times, letting the numbers play out in
slightly different ways each time. For each simulation, sprint velocity and backlog changes
are assumed based on how they varied in previous sprints, producing one possible path from start to finish.

By repeating this thousands of times, we can explore the many ways the project might unfold
and see how much the results can differ.

This approach, called Monte Carlo simulation, turns a single estimate into a richer picture
of what to expect - showing not just one possible completion date, but the range of outcomes
that could realistically occur.
"""

st.subheader("Model")
r"""
Let's model the project as a sequence of steps - sprints - each of which reduces the backlog size
based on what the team completes (`v`) and the changes to the backlog (`d`) such as additions,
removals, or re-estimations.

By simulating different sequences of `v` and `d`, we can see how their interaction shapes
the project's trajectory over time.

To keep the simulation manageable, we make a few simplifying assumptions:

* `v` and `d` in every sprint are treated as independent.
* `v` and `d` follow patterns already observed in the data
  - we assume they are normally distributed
* Other factors (scope changes, new feature requests, team capacity variations, etc.)
  are considered implicitly embedded in the data
  - assumed roughly uniform across the project

These assumptions reduce the system to a computable model while still capturing the key quantitative
drivers of project duration.
"""
st.caption("""
Team velocity can reasonably be approximated as normally distributed: group performance tends
to bounce around a median value with natural variation.

Backlog changes are different — they aggregate many factors:

* Estimates naturally evolve
* Requirements and scope creep
* New features, bugs, and technical debt

In reality, backlog changes are unlikely to be perfectly normal; they may be log-normal
or have fat tails.

We also assume their magnitude is smaller than team velocity to prevent
the project from running indefenitely.
""")

st.subheader("Simulation Code")

with st.echo():
    NUMBER_OF_SIMULATIONS = 4096

    @st.cache_data
    def run_simulations(n: int) -> Sequence[ProjectSimResult]:
        fn = partial(simulate_project, project=project, base_sprints=5, max_sprints=100)
        return [fn(rng_seed=i) for i in range(n)]

    sim_results = run_simulations(NUMBER_OF_SIMULATIONS)
    sim_results_df = pd.DataFrame(sim_results)

count_not_completed = (~sim_results_df["completed"]).sum()
if count_not_completed > 0:
    st.error(f"Not completed count: {count_not_completed}")

st.subheader("Simulation Results 📊")
rf"""
After running the project simulation `{NUMBER_OF_SIMULATIONS}` times, we can see not just an average,
but the full spread of what might happen. Let's take a closer look at the results.
"""
sim_results_describe = sim_results_df["duration"].describe(percentiles=[0.5, 0.75, 0.85, 0.9, 0.99]).to_frame().T
r"""Simulations statistics:"""
st.dataframe(sim_results_describe[["count", "mean", "std", "min", "max"]])
r"""Duration percentiles:"""
st.dataframe(sim_results_describe[["50%", "75%", "85%", "90%", "99%"]].rename_axis("Percentile"))

r"""
* With an average project duration of `33` sprints, our simulations suggest a standard deviation
of `4.4` sprints. That translates to a window of uncertainty of roughly `9` sprints -
the project could realistically finish much earlier or later than expected. If
a sprint lasts two weeks, that's more than **four months of potential wiggle**,
enough to surprise even the most confident stakeholders.
* Some simulations finish in just `21` sprints 🚀. In a real project, this is highly unlikely -
definitely not something to rely on — but theoretically, it's possible.
* The average of `33` sprints also means that half of the simulations take longer than that.
* The `85%` or `90%` percentiles are more practical for stakeholder communication.
Think of a percentile as a confidence level: for example,
“We are `85%` confident the team can complete the project in `38` sprints.”
"""

r"""
Remember our initial naive estimate was `33` sprints. So why does our model predict `38` instead?

The first estimate only looked at averages, assuming the team would hit the same net velocity
every sprint. Reality is messier: some sprints are faster, some slower, and the backlog keeps
shifting. Running thousands of simulated project paths shows that slower sequences matter -
a few dips here and there add up, pushing likely completion further out.
"""
with st.container(border=True):
    r"""
    ✨ :blue[Instead of a single date, give stakeholders a range with confidence levels.
    For example: “We are 85% confident the team will finish within 38 sprints.”]
    """
r"""
The `38`-sprint figure isn't just a pessimistic forecast; it's a realistic outcome that reflects
the ups and downs, and a timeline the team can actually commit to without expecting excessive
overtime or long Fridays.
"""


st.subheader("Simulation Results - Details 🔍")

rf"""
Initial data for all simulation runs was the same:
* Initial Backlog Size: `{BACKLOG_INITIAL_SIZE}`
* Completed sprints: `{CURRENT_SPRINT}`
* Velocity Avg / Std: `{v_mean:.2f}` / `{v_std:.2f}`
* Backlog Delta Avg / Std: `{d_mean:.2f}` / `{d_std:.2f}`

Let's pick few project paths and plot them to see what is actually happening there.
"""

ranges = [range(20, 24), range(32, 36), range(37, 41), range(49, 53)]
chosen_sims: list[ProjectSimResult | None] = [None] * len(ranges)
for sim in sim_results:
    for i, r in enumerate(ranges):
        if chosen_sims[i] is None and sim.duration in r and sim.completed:
            chosen_sims[i] = sim
    if all(chosen_sims):
        break

chosen_sims_df = pd.DataFrame(chosen_sims)
chosen_sims_df["v_mean"] = chosen_sims_df["vs"].apply(np.mean)
chosen_sims_df["v_std"] = chosen_sims_df["vs"].apply(np.std)
chosen_sims_df["d_mean"] = chosen_sims_df["ds"].apply(np.mean)
chosen_sims_df["d_std"] = chosen_sims_df["ds"].apply(np.std)
chosen_sims_df = chosen_sims_df.drop(columns=["vs", "ds", "bs", "completed"])

st.dataframe(chosen_sims_df)

with st.container(horizontal=True):
    for sim in chosen_sims:
        if sim is None:
            continue
        fig = prj_chart_backlog_trajectory(bs=sim.bs, vs=sim.vs, ds=sim.ds, chart_subtitle=f"Duration: {sim.duration}")
        st.pyplot(fig)

rf"""
Let's interpret the results.

Across all four cases, team velocity is relatively stable, with moderate fluctuations. As expected,
it is slightly higher in the faster scenarios and slightly lower in the slower ones - the gap between
the fastest and slowest case is about `1` story point per sprint. That's exactly how we modeled it.

The real difference shows up in backlog changes.

In the fastest case, backlog delta swings widely but additions are balanced by aggressive removals.
Scope is actively managed. The backlog is volatile, yet it doesn't drift upward. Over time, the average
delta stays low, and the team keeps gaining ground.

In the slower cases, volatility plays out differently. The realized average backlog delta ends up higher
than the model parameter (`{d_mean:.2f}` vs `20.25` vs `25.65`). Statistically, this is simply variance at work -
on a finite path, the observed mean does not have to converge to the expected one. But in practice,
it illustrates that scope growth which slightly outpaces delivery for long enough stretches the timeline.

Velocity barely changes across scenarios. What truly separates a 23-sprint project from a 49-sprint one
is how backlog volatility resolves over time - whether fluctuations cancel out, or gradually accumulate.
"""

st.subheader("Is Project Duration Normally Distributed?")

r"""
In our model, each sprint is driven by two random components: team velocity (`v`) and backlog change (`d`).
For simplicity, we assumed both follow normal distributions (remember, in reality they probably are not
normal, or at least backlog changes is not), based on patterns observed in historical data. Each sprint
reduces (or increases) the backlog according to the net effect of these two variables, and the project
finishes when the backlog reaches zero.

Even though velocity and backlog changes are modeled as normally distributed, we should not expect project
duration to follow a normal distribution. Duration emerges from a discrete sequence of cumulative sprint
outcomes and ends when the backlog reaches zero - a stopping-time process rather than a simple draw from
a distribution.

This structure naturally introduces asymmetry:

* finishing much earlier than average requires consistently strong progress,
* while finishing later can result from just a few unfavorable streaks ☔.

As a result, we should expect a distribution that is roughly bell-shaped near the center but slightly right-skewed,
with a longer upper tail.

> Unlike continuous approximations of first-passage times, the discreteness of sprints and the combined
> randomness of velocity and backlog changes make this distribution unique - it does not follow a standard
> inverse Gaussian or any simple closed form.

Following histogram chart illustrates this point.
"""

durations = sim_results_df["duration"].to_numpy()
fig, ax1, ax2 = chart_distribution_histogram(durations)

params = invgauss.fit(durations)
xs = np.linspace(durations.min(), durations.max(), 500)
ys_invg_pdf = invgauss.pdf(xs, *params)
invg_mode = xs[np.argmax(ys_invg_pdf)]
ax1.plot(xs, ys_invg_pdf, color="green", label="Inverse Gaussian Distribution")
# ? inv g cdf: ax2.plot(xs, invgauss.cdf(xs, *params), color="green", label="Cumulative Distribution (Inv G)", linestyle="--")  # noqa: E501
ax1.axvline(invg_mode, color="green", linestyle="--", label=f"Mode (Inv G): {invg_mode:.2f}", lw=0.5)
fig.legend(loc="upper left")

st.pyplot(fig)

st.subheader("Interactive Probability Histogram 📈")
r"""
Even a simple probability distribution histogram can be a powerful communication tool.
By interacting with it, you can pick a confidence level and immediately see the range
of likely project durations, making uncertainty tangible for stakeholders.
"""


@st.fragment
def fr_histogram_interactive(ds: np.ndarray) -> None:
    conf = st.slider("Confidence Level", min_value=55, max_value=95, value=85, step=5, format="%i%%") / 100
    bins = np.arange(ds.min(), ds.max() + 2)  # +2 to include the last edge
    hist, edges = np.histogram(ds, bins=bins)
    cdf_model = np.cumsum(hist) / hist.sum()
    idx = np.argmax(cdf_model > conf)
    d = edges[idx]
    mu, sigma = np.mean(ds), np.std(ds, ddof=0)

    fig, ax1, ax2 = chart_distribution_histogram(durations)
    ax2.axhline(conf, color="green", linestyle="--", label=f"Confidence Level: {conf:.2f}%")
    ax2.axvline(d - 1, color="green", linestyle="--", linewidth=0.7)
    ax2.axvline(d, color="green", linestyle="--", label=f"Likely Duration: {d}")
    ax2.scatter(d - 0.5, conf, facecolors="None", edgecolor="green", marker="o", s=64)
    ax2.annotate(
        f"{conf:.0%} confidence duration is {d}",
        xy=(d - 0.5, conf),
        xytext=(float(mu + 1.5 * sigma), conf - 0.15),
        arrowprops={"arrowstyle": "->"},
    )
    fig.legend(loc="upper left")
    st.pyplot(fig)


fr_histogram_interactive(durations)

st.header("Learning from Others 📚")
r"""
The concept, I explored in this article - focusing on ranges, confidence levels, and simulated project paths -
isn't yet mainstream in project management discussions especially in context of software project.

It draws inspiration from a variety of sources.

One of the sparks for this article came from John Hull's *Options, Futures, and Other Derivatives*,
which made me think about the parallels between project management and finance - and how we can frame
uncertainty in a similar way.

Classic estimation ideas, like **3-point estimation** and **PERT**, along with practical guides such as
Steve McConnell's *Software Estimation*, also shaped my thinking along the way.

I'm grateful to the many authors, known and unknown, whose work help me shaped these ideas.
"""

with st.container(border=True):
    r"""
    :blue[Estimating project timelines is never exact, but understanding the range of likely outcomes helps
    teams and stakeholders plan more realistically. By focusing on probabilities and confidence levels,
    we replace false certainty with informed decision-making.]
    """

st.header("See Also")
st.page_link("pages/page_01_inv_gauss.py")
