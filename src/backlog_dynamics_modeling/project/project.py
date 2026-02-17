from dataclasses import KW_ONLY, dataclass, field

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


@dataclass(slots=True, frozen=True, kw_only=True)
class Project:
    name: str
    backlog_initial_size: int
    sprint_team_velocity: list[int] = field(default_factory=list[int], init=False)
    sprint_backlog_delta: list[int] = field(default_factory=list[int], init=False)

    def add_sprints_data(
        self,
        data: pd.DataFrame,
        *,
        col_sprint: str = "sprint",
        col_velocity: str = "v",
        col_b_delta: str = "d",
    ) -> None:
        if not np.array_equal(data[col_sprint], np.arange(len(data))):
            msg = "Sprints data must be indexed by integers starting from 0."
            raise ValueError(msg)
        if self.sprint_team_velocity or self.sprint_backlog_delta:
            msg = "Sprints data must be empty."
            raise ValueError(msg)
        self.sprint_team_velocity.extend(data[col_velocity])
        self.sprint_backlog_delta.extend(data[col_b_delta])


@dataclass(slots=True, frozen=True)
class DeterministicProjectEstimator:
    project: Project
    _: KW_ONLY
    base_sprints_number: int
    max_sprints_number: int

    def team_velocity(self) -> np.ndarray:
        n = min(self.base_sprints_number, self.max_sprints_number, len(self.project.sprint_team_velocity))
        n_max = self.max_sprints_number
        v_mean = np.mean(self.project.sprint_team_velocity[:n])
        vs = np.full((n_max,), v_mean)
        vs[0:n] = self.project.sprint_team_velocity[:n]
        return vs

    def backlog_size(self) -> np.ndarray:
        n = min(self.base_sprints_number, self.max_sprints_number, len(self.project.sprint_backlog_delta))
        n_max = self.max_sprints_number
        d_mean = np.mean(self.project.sprint_backlog_delta[:n])

        ds = np.full((n_max,), d_mean)
        ds[0:n] = self.project.sprint_backlog_delta[:n]
        ds[0] += self.project.backlog_initial_size
        b = (ds - self.team_velocity()).cumsum()
        b[b < 0] = 0

        return b


@dataclass(slots=True, frozen=True, kw_only=True)
class ProjectSimResult:
    completed: bool
    duration: int
    vs: np.ndarray
    ds: np.ndarray
    bs: np.ndarray


def simulate_project(
    project: Project,
    *,
    rng_seed: int,
    base_sprints: int,
    max_sprints: int,
    ddof: int = 1,
) -> ProjectSimResult:
    rng = np.random.default_rng(rng_seed)

    vs = []
    ds = []
    bs = []
    backlog_size = project.backlog_initial_size

    for i in range(base_sprints):
        v = project.sprint_team_velocity[i]
        d = project.sprint_backlog_delta[i]
        backlog_size += d - v
        vs.append(v)
        ds.append(d)
        bs.append(backlog_size)

    v_mean = np.mean(vs)
    v_std = np.std(vs, ddof=ddof)
    d_mean = np.mean(ds)
    d_std = np.std(ds, ddof=ddof)

    if v_mean <= d_mean:
        msg = "Unstable system: expected backlog drift is non-negative."
        raise ValueError(msg)

    for _ in range(base_sprints, max_sprints):
        v = max(0, rng.normal(v_mean, v_std))
        d = rng.normal(d_mean, d_std * 2)
        backlog_size += d - v
        vs.append(v)
        ds.append(d)
        bs.append(backlog_size)
        if backlog_size <= 0:
            break

    return ProjectSimResult(
        completed=backlog_size <= 0,
        duration=len(vs),
        vs=np.array(vs),
        ds=np.array(ds),
        bs=np.array(bs),
    )
