from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True, kw_only=True)
class Project:
    name: str
    backlog_initial_size: int
    sprint_team_velocity: list[int] = field(default_factory=list[int], init=False)
    sprint_backlog_delta: list[int] = field(default_factory=list[int], init=False)
