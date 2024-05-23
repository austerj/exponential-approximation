import math
import typing
from abc import ABC, abstractmethod

import mpmath

from expapprox import errors

# mpmath float alias - actual type is dynamic and not handled properly by pyright etc.
mpf = float


class Approximator(ABC):
    """Base class for approximators."""

    __slots__ = ()

    def _fields(self) -> list[str]:
        """Get fields used in string representation."""
        return [""]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self._fields())})"

    @abstractmethod
    def __call__(self, x: int | float) -> mpf:
        """Approximate function value as mpmath float."""
        raise NotImplementedError

    def try_call(self, x: int | float) -> mpf:
        """Return function value approximation as mpmath float or NaN on error."""
        try:
            return self(x)
        except:
            return mpmath.nan

    @classmethod
    @abstractmethod
    def ref(cls, x: float) -> mpf:
        """Compute reference value."""
        raise NotImplementedError

    def benchmark(self, xs: typing.Sequence[float]) -> list[float]:
        """Compute relative errors (from reference values) for sequence of inputs."""
        return [relative_error(self.try_call(x), self.ref(x)) for x in xs]


class FixedPointApproximator(Approximator, ABC):
    """Base class for fixed-point number approximators."""

    __slots__ = ("decimals", "identity")

    def __init__(self, decimals: int):
        if decimals < 0:
            raise errors.InvalidDecimalsError()
        # fixed-point decimals
        self.decimals = decimals
        # multiplicative identity of fixed-point specification
        self.identity = 10**self.decimals

    def _fields(self) -> list[str]:
        return [f"decimals={self.decimals}"]

    def __call__(self, x: int | float) -> mpf:
        return self.to_float(self.approx(self.to_fixed(x)))

    @property
    def workdps(self):
        """Context manager for mpmath decimals."""
        return mpmath.workdps(self.decimals)

    @abstractmethod
    def approx(self, x: int) -> int:
        """Approximate function value from fixed-point number as fixed-point number."""
        raise NotImplementedError

    def to_float(self, x: int) -> mpf:
        """Convert fixed-point number to mpmath float."""
        return mpmath.mpf(x) / self.identity

    def to_fixed(self, x: int | float) -> int:
        """Convert number to fixed-point representation."""
        # using mpmath float for intermediary multiplication before flooring
        return math.floor(mpmath.mpf(x) * self.identity)

    def benchmark(self, xs: typing.Sequence[float]) -> list[float]:
        with self.workdps:
            return super().benchmark(xs)


class ExponentialApproximator(FixedPointApproximator, ABC):
    """Base class for fixed-point approximator of the exponential function."""

    __slots__ = ()

    @classmethod
    def ref(cls, x: float) -> mpf:
        return mpmath.exp(x)


def relative_error(approx: mpf, ref: mpf) -> float:
    """Compute the relative error for an approximation compared to a reference value."""
    # handle NaN values
    if not (math.isfinite(approx) and math.isfinite(ref)):
        return math.nan
    # handle zero-division
    if ref == 0.0:
        return 0.0 if approx == ref else math.inf
    # convert (mpmath float) result to built-in float
    return float(abs((approx - ref) / ref))
