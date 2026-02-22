from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING, Any, Final
from warnings import deprecated

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm

from backlog_dynamics_modeling.project.pdpd_model import (
    PDProbDModel,
    PDProbDModelContinuous,
    PDProbDModelDiscrete,
    PDProbDModelInvGauss,
    PDProbDModelNormal,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class PDProbModelPlotType(Flag):
    PDF = auto()
    CDF = auto()
    MODE = auto()
    MEAN = auto()
    MEDIAN = auto()

    ALL = PDF | CDF | MODE | MEDIAN | MEAN
    VMARKERS = MODE | MEDIAN | MEAN

    def plot_kwargs(self) -> Mapping[str, Any]:
        return _LINESTYLES[self]

    def supported_by_model(self, model: PDProbDModel) -> bool:
        t = next(i for i in type(model).mro() if i in _MODEL_SUP_PLOT_TYPES)
        return t and self in _MODEL_SUP_PLOT_TYPES[t]

    def label_prefix(self) -> str:
        return _PLOT_TYPE_LABEL_PREFIX[self]

    def value_from_model(self, model: PDProbDModel) -> float:
        match self:
            case PDProbModelPlotType.MODE:
                return model.mode
            case PDProbModelPlotType.MEDIAN:
                return model.median
            case PDProbModelPlotType.MEAN:
                return model.mean
            case _:
                msg = f"Unsupported plot type: {self}"
                raise ValueError(msg)


_PLOT_TYPE_LABEL_PREFIX: Final[Mapping[PDProbModelPlotType, str]] = {
    PDProbModelPlotType.MODE: "Mode",
    PDProbModelPlotType.MEDIAN: "Median",
    PDProbModelPlotType.MEAN: "Mean",
}

_LINESTYLES: Final[Mapping[PDProbModelPlotType, Mapping[str, Any]]] = {
    PDProbModelPlotType.PDF: {
        "linestyle": "-",
        "alpha": 0.55,
        "linewidth": 1.0,
    },
    PDProbModelPlotType.CDF: {
        "linewidth": 1.0,
    },
    PDProbModelPlotType.MEAN: {
        "linestyle": ":",
        "linewidth": 0.85,
    },
    PDProbModelPlotType.MEDIAN: {
        "linestyle": "--",
        "linewidth": 0.75,
    },
    PDProbModelPlotType.MODE: {
        "linestyle": "-",
        "linewidth": 0.75,
    },
}

_MODEL_SUP_PLOT_TYPES: Final[Mapping[type[PDProbDModel], PDProbModelPlotType]] = {
    PDProbDModelDiscrete: PDProbModelPlotType.ALL,
    PDProbDModelContinuous: PDProbModelPlotType.ALL & ~PDProbModelPlotType.MODE,
}


def _visualize_vmarkers(ax: Axes, model: PDProbDModel, *, color: str, types: PDProbModelPlotType) -> None:
    types = types & PDProbModelPlotType.VMARKERS
    for t in types:
        if t.supported_by_model(model):
            value = t.value_from_model(model)
            kw = t.plot_kwargs()
            label = f"{t.label_prefix()} ({model.label}): {value:.2f}"
            ax.axvline(value, color=color, label=label, **kw)


class ChartProjectDurationDistribution:
    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (8, 4),
        title: str = "Project Duration Distribution",
        x_min: float = 0,
        x_max: float = 100,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.fig, self.axf = plt.subplots(figsize=figsize)
        self.axp = self.axf.twinx()
        self.fig.suptitle(title)
        self.axf.set_xlabel("Project Duration (sprints)")
        self.axf.set_ylabel("Frequency")
        self.axp.set_ylabel("Probability")
        self.axf.grid(visible=True, linestyle="--", alpha=0.3, axis="x")
        self.axp.grid(visible=True, linestyle="--", alpha=0.3, axis="y")
        self.axf.axhline(0, color="black", alpha=0.4, linewidth=0.5)

    def get_figure(self) -> tuple[Figure, Axes, Axes]:
        self.fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.4))
        return self.fig, self.axf, self.axp

    def plot_model(self, model: PDProbDModel, *, color: str, types: PDProbModelPlotType | None = None) -> None:
        types = types or PDProbModelPlotType.PDF
        match model:
            case PDProbDModelDiscrete():
                self._plot_model_discrete(model, color=color, types=types)
            case PDProbDModelNormal() | PDProbDModelInvGauss():
                self._plot_model_continuous(model, color=color, types=types)
        _visualize_vmarkers(self.axf, model, color=color, types=types)

    def _plot_model_discrete(self, model: PDProbDModelDiscrete, *, color: str, types: PDProbModelPlotType) -> None:
        ds = model.ds
        d_min, d_max = ds.min(), ds.max()
        if types & PDProbModelPlotType.PDF:
            bins = np.arange(d_min, d_max + 1)
            kw = PDProbModelPlotType.PDF.plot_kwargs()  # _LINESTYLES[PDProbModelPlotType.PDF]
            self.axf.hist(ds, color=color, density=True, bins=bins, **kw, label=f"Prob. dist. ({model.label})")
        if types & PDProbModelPlotType.CDF:
            bins = np.arange(d_min, d_max + 2)
            hist, edges = np.histogram(ds, bins=bins)
            cdf_model = np.cumsum(hist) / hist.sum()
            kw = PDProbModelPlotType.CDF.plot_kwargs()
            self.axp.plot(edges[:-1], cdf_model, color=color, label=f"Cum. Prob. dist. ({model.label})", **kw)

    def _plot_model_continuous(self, model: PDProbDModelContinuous, *, color: str, types: PDProbModelPlotType) -> None:
        xs = np.linspace(self.x_min, self.x_max, 500)
        if types & PDProbModelPlotType.PDF:
            kw = _LINESTYLES[PDProbModelPlotType.PDF]
            self.axf.plot(xs, model.pdf(xs), color=color, label=f"Prob. dist. ({model.label})", **kw)
        if types & PDProbModelPlotType.CDF:
            kw = _LINESTYLES[PDProbModelPlotType.CDF]
            self.axp.plot(xs, model.cdf(xs), color=color, label=f"Cum. Prob. dist. ({model.label})", **kw)


# TODO (ihor): replace with ChartProjectDurationDistribution
@deprecated("Use ChartProjectDurationDistribution instead")
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
    ys_norm = normdist.pdf(xs)  # pyright: ignore[reportAttributeAccessIssue]
    ax1.plot(xs, ys_norm, color="black", lw=2, label="Normal Distribution")
    ax1.axvline(float(mu), color="orange", linestyle="--", label=f"Mean: {mu:.2f}", lw=0.5)
    ax1.axvline(mode, color="red", linestyle="--", label=f"Mode: {mode}", lw=0.5)

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

    return fig, ax1, ax2
