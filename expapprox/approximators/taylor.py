import math

from expapprox import errors
from expapprox.approximator import FixedPointExponentialApproximator


class TaylorApproximator(FixedPointExponentialApproximator):
    """Order-N Taylor fixed-point approximator of the exponential function."""

    __slots__ = ("order", "factorial", "constants")

    def __init__(self, decimals: int, order: int):
        super().__init__(decimals)
        if order < 1:
            raise errors.ApproximatorError("Invalid order {order}; must be 1 or greater")
        self.order = order
        self.constants = [self.to_fixed(math.factorial(order) / math.factorial(i)) for i in reversed(range(order))]
        self.factorial = math.factorial(order)

    def _fields(self):
        return [*super()._fields(), f"order={self.order}"]

    def approx(self, x: int) -> int:
        # initialize accumulator to N! + x (in fixed-point representation)
        accumulator = x.__class__(self.constants[0])
        accumulator += x
        # accumulate Horner terms
        for constant in self.constants[1:]:
            # multiply to get next order and rescale
            accumulator *= x
            accumulator //= self.identity
            # add constant
            accumulator += constant
        return accumulator // self.factorial
