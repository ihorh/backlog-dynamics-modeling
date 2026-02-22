from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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
