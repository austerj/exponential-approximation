import math
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import mpmath

# mpmath float alias - actual type is dynamic and not handled properly by pyright etc.
mpf = float


@dataclass(frozen=True, slots=True)
class FixedPointApproximator(ABC):
    """Base class for fixed-point number approximators."""

    # fixed-point specification
    decimals: int
    _identity: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "_identity", 10**self.decimals)

    def __call__(self, x: int) -> mpf:
        """Approximate function value from fixed-point number as mpmath float."""
        return self.to_float(self.approx(x))

    @property
    def identity(self) -> int:
        """Multiplicative identity of fixed-point specification."""
        return self._identity

    @property
    def workdps(self):
        """Context manager for mpmath decimals."""
        return mpmath.workdps(self.decimals)

    @abstractmethod
    def approx(self, x: int) -> int:
        """Approximate function value from fixed-point number as fixed-point number."""
        raise NotImplementedError

    @abstractmethod
    def ref(self, x: int) -> mpf:
        """Compute reference value from fixed-point number as mpmath float."""
        raise NotImplementedError

    def to_float(self, x: int) -> mpf:
        """Convert fixed-point number to mpmath float."""
        return mpmath.mpf(x) / self.identity

    def to_fixed(self, x: float | int) -> int:
        """Convert number to fixed-point representation."""
        # using mpmath float for intermediary multiplication before flooring
        return math.floor(mpmath.mpf(x) * self.identity)

    def benchmark(self, xs: typing.Sequence[float]) -> list[float]:
        """Compute relative errors (from reference values) for sequence of inputs."""
        return [relative_error(self(x), self.ref(x)) for x in (self.to_fixed(x) for x in xs)]


def relative_error(approx: mpf, ref: mpf) -> float:
    """Compute the relative error for an approximation compared to a reference value."""
    # handle zero-division
    if ref == 0.0:
        return 0.0 if approx == ref else math.inf
    # force result to built-in float
    return float(abs((approx - ref) / ref))
