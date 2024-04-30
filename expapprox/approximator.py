import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FixedPointApproximator(ABC):
    """Base class for fixed-point number approximators."""

    decimals: int

    @abstractmethod
    def __call__(self, x: int) -> int:
        """Approximate function value."""
        raise NotImplementedError

    @abstractmethod
    def ref(self, x: int) -> float:
        """Get reference value used for benchmarking accuracy of approximator."""
        raise NotImplementedError

    def to_float(self, x: int):
        """Convert fixed-point number to float."""
        return x / 10**self.decimals

    def to_fixed(self, x: int | float):
        """Convert number to fixed-point representation."""
        return math.floor(x * 10**self.decimals)
