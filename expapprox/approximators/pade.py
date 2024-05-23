import math

from expapprox import errors
from expapprox.approximator import FixedPointExponentialApproximator
from expapprox.approximators.bshift import BitShiftApproximator


class PadeApproximator(FixedPointExponentialApproximator):
    """Order-[N/N] Padé fixed-point approximator of the exponential function."""

    __slots__ = ("order", "coefficients", "constant")

    def __init__(self, decimals: int, order: int):
        super().__init__(decimals)
        if order < 1:
            raise errors.ApproximatorError("Invalid order {order}; must be 1 or greater")
        self.order = order
        self.coefficients = coefficients(order)
        self.constant = self.coefficients[0] * self.identity

    def _fields(self):
        return [*super()._fields(), f"order={self.order}"]

    def approx(self, x: int) -> int:
        # initialize even accumulator to c_0 (constant term)
        even_accumulator = x.__class__(self.constant)
        # initialize odd accumulator to c_1 * x
        odd_accumulator = x
        odd_accumulator *= self.coefficients[1]
        # accumulate even- and odd power terms
        # NOTE: can (should!) be unrolled in practice (where order etc. is fixed)
        x_pow = x
        for i, c in enumerate(self.coefficients[2:]):
            # multiply to get next order and rescale
            x_pow *= x
            x_pow //= self.identity
            # compute term and add to corresponding accumulator
            x_term = x_pow
            # NOTE: skipping final coefficient (== 1) can be hardcoded for fixed configuration
            if i < self.order - 2:
                x_term *= c
            # NOTE: even/odd branching can be hardcoded upon loop unrolling for fixed configuration
            if i % 2:
                odd_accumulator += x_term
            else:
                even_accumulator += x_term
        # validate non-zero denominator
        if even_accumulator <= odd_accumulator:
            raise errors.ApproximatorDomainError("Exceeded critical point")
        # compute numerator and denominator from accumulators
        numerator = even_accumulator + odd_accumulator
        denominator = even_accumulator - odd_accumulator
        # rescale numerator for fixed-point division
        numerator *= self.identity
        return numerator // denominator


def coefficients(order: int) -> list[int]:
    """Compute the Padé[N/N] coefficients of a given order for the exponential function."""
    return [math.factorial(2 * order - n) // (math.factorial(n) * math.factorial(order - n)) for n in range(order + 1)]


class BitShiftPadeApproximator(BitShiftApproximator):
    """Bit-shifted order-[N/N] Padé fixed-point approximator of the exponential function."""

    def __init__(self, decimals: int, order: int):
        super().__init__(decimals)
        self.remainder_approximator = PadeApproximator(decimals, order)
