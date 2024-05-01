import math
import typing
from abc import ABC, abstractmethod

import mpmath

# mpmath float alias - actual type is dynamic and not handled properly by pyright etc.
mpf = float


class FixedPointApproximator(ABC):
    """Base class for fixed-point number approximators."""

    __slots__ = ("decimals", "identity")

    def __init__(self, decimals: int):
        # fixed-point decimals
        self.decimals = decimals
        # multiplicative identity of fixed-point specification
        self.identity = 10**self.decimals

    def _fields(self):
        """Get fields used in string representation."""
        return [f"decimals={self.decimals}"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self._fields())})"

    def __call__(self, x: int) -> mpf:
        """Approximate function value from fixed-point number as mpmath float."""
        return self.to_float(self.approx(x))

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


class ExponentialApproximator(FixedPointApproximator, ABC):
    """Base class for fixed-point approximator of the exponential function."""

    def ref(self, x: int) -> mpf:
        return mpmath.exp(self.to_float(x))


def relative_error(approx: mpf, ref: mpf) -> float:
    """Compute the relative error for an approximation compared to a reference value."""
    # handle zero-division
    if ref == 0.0:
        return 0.0 if approx == ref else math.inf
    # convert (mpmath float) result to built-in float
    return float(abs((approx - ref) / ref))
