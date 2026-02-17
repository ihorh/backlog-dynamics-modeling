import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from scipy.stats import norm


def prj_chart_backlog_trajectory(
    *,
    vs: np.ndarray | None = None,
    ds: np.ndarray | None = None,
    bs: np.ndarray,
    project_name: str | None = None,
    chart_subtitle: str | None = None,
) -> Figure:
    project_name = project_name or ""
    chart_subtitle = chart_subtitle or ""

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.set_xlabel("Sprint")
    ax1.set_ylabel("Backlog Size")
    ax1.grid()

    if vs is not None or ds is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Team Velocity / Backlog Delta")

    if vs is not None:
        ax2.plot(vs, color="green", label="Velocity")
        ax2.axhline(vs.mean(), color="green", linestyle=":", label=f"Avg Velocity: {vs.mean():.2f}")

    if ds is not None:
        ax2.plot(ds, color="orange", label="Backlog Delta")
        ax2.axhline(0, color="pink")
        ax2.axhline(ds.mean(), color="orange", linestyle=":", label=f"Avg Backlog Delta: {ds.mean():.2f}")

    # * ax1 - backlog size
    ax1.plot(bs, color="red", label="Remaining Backlog Size")

    if t := np.argmax(bs <= 0):
        ax1.axvline(float(t), color="green", linestyle=":", label=f"Done in sprint {t}")

    fig.legend(bbox_to_anchor=(0.9, 0.9), loc="upper right")

    fig.suptitle(f"Project Trajectory: {project_name}\n{chart_subtitle}")

    return fig


def chart_distribution_histogram(ds: np.ndarray) -> tuple[Figure, Axes, Axes]:
    d_min, d_max = ds.min(), ds.max()
    xs = np.linspace(d_min, d_max, 500)
    mu, sigma = np.mean(ds), np.std(ds, ddof=0)
    mode = stats.mode(ds)[0]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    fig.suptitle("Monte Carlo Distribution of Project Duration")
    ax1.hist(
        ds,
        color="indigo",
        alpha=0.45,
        density=True,
        bins=np.arange(d_min, d_max + 1),
    )
    ax1.xaxis.set_label_text("Project Duration (sprints)")
    ax1.yaxis.set_label_text("Frequency")
    ax1.grid(visible=True, linestyle="--", alpha=0.3)

    normdist = norm(mu, sigma)
    ax1.plot(xs, normdist.pdf(xs), color="black", lw=2, label="Normal Distribution")
    ax1.axvline(mu, color="orange", linestyle="--", label=f"Mean: {mu:.2f}", lw=0.5)
    ax1.axvline(mode + 0.5, color="red", linestyle="--", label=f"Mode: {mode}", lw=0.5)

    # * cumulative probability distributions
    ax2 = ax1.twinx()
    ax2.yaxis.set_label_text("Probability")
    # * normal cumulative probability distribution
    ax2.plot(xs, normdist.cdf(xs), color="pink", lw=1, label="Cumulative Distribution (normal)")
    # * Empirical CDF from simulations
    bins = np.arange(ds.min(), ds.max() + 2)  # +2 to include the last edge
    hist, edges = np.histogram(ds, bins=bins)
    cdf_model = np.cumsum(hist) / hist.sum()
    ax2.plot(edges[:-1], cdf_model, color="red", lw=1, label="Cumulative Distribution (model)")

    fig.legend(loc="upper left")
    return fig, ax1, ax2
