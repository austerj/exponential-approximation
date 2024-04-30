import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FixedPointApproximator(ABC):
    """Base class for fixed-point number approximators."""

    # fixed-point specification
    decimals: int
    _identity: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "_identity", 10**self.decimals)

    @property
    def identity(self) -> int:
        """Multiplicative identity of fixed-point specification."""
        return self._identity

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
        return x / self.identity

    def to_fixed(self, x: int | float):
        """Convert number to fixed-point representation."""
        return math.floor(x * self.identity)
