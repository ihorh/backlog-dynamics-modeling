from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, override

import numpy as np
from scipy.stats import invgauss, norm

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass(slots=True, frozen=True, kw_only=True)
class PDProbDModel:
    label: str

    @property
    def mean(self) -> float: ...
    @property
    def median(self) -> float: ...
    @property
    def mode(self) -> float: ...


class ContinuousDistribution(Protocol):
    def mean(self) -> float: ...
    def median(self) -> float: ...
    def pdf(self, xs: ArrayLike) -> np.ndarray: ...
    def cdf(self, xs: ArrayLike) -> np.ndarray: ...


@dataclass(frozen=True, kw_only=True)
class PDProbDModelDiscrete(PDProbDModel):
    ds: np.ndarray

    def __post_init__(self) -> None:
        self.ds.flags.writeable = False

    @cached_property
    def mean(self) -> float:
        return self.ds.mean()

    @cached_property
    def median(self) -> float:
        return np.median(self.ds)

    @cached_property
    def mode(self) -> float:
        values, counts = np.unique(self.ds, return_counts=True)
        return values[np.argmax(counts)]


class PDProbDModelContinuous(PDProbDModel):
    model: ContinuousDistribution = field(init=False)

    @property
    @override
    def median(self) -> float:
        return self.model.median()

    @property
    @override
    def mean(self) -> float:
        return self.model.mean()

    def pdf(self, xs: ArrayLike) -> np.ndarray:
        return self.model.pdf(xs)

    def cdf(self, xs: ArrayLike) -> np.ndarray:
        return self.model.cdf(xs)


@dataclass(slots=True, frozen=True, kw_only=True)
class PDProbDModelNormal(PDProbDModelContinuous):
    mean: float
    std: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", norm(loc=self.mean, scale=self.std))


@dataclass(slots=True, frozen=True, kw_only=True)
class PDProbDModelInvGauss(PDProbDModelContinuous):
    mu: float
    loc: float
    scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", invgauss(mu=self.mu, loc=self.loc, scale=self.scale))

    @staticmethod
    def first_passage_time(*, a: float, mean: float, std: float, label: str) -> PDProbDModelInvGauss:
        mu_ig = a / mean
        lambda_ig = (a**2) / (std**2)
        return PDProbDModelInvGauss.from_standard_params(label=label, mean=mu_ig, shape=lambda_ig)


    @staticmethod
    def fit_data(data: np.ndarray, *, label: str) -> PDProbDModelInvGauss:
        mu, loc, scale = invgauss.fit(data)
        return PDProbDModelInvGauss(label=label, mu=mu, loc=loc, scale=scale)

    @staticmethod
    def from_standard_params(*, label: str, mean: float, shape: float) -> PDProbDModelInvGauss:
        mu = mean / shape
        scale = shape
        loc = 0
        return PDProbDModelInvGauss(label=label, mu=mu, loc=loc, scale=scale)
