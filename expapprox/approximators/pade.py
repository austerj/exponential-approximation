import math

from expapprox import errors
from expapprox.approximator import ExponentialApproximator
from expapprox.approximators.bshift import BitShiftApproximator


class PadeApproximator(ExponentialApproximator):
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
        # precompute power terms from ABSOLUTE x values (in fixed-point representation)
        x_terms = self.terms(abs(x))
        # compute even and odd term sums
        even_sum = sum(x_terms[::2])
        odd_sum = sum(x_terms[1::2])
        # compute numerator and denominator from term sums
        numerator = even_sum + odd_sum
        denominator = even_sum - odd_sum
        if x <= 0:
            # invert numerator and denominator for negative x
            return (denominator * self.identity) // numerator
        else:
            if denominator <= 0:
                raise errors.ApproximatorDomainError("Exceeded critical point")
            return (numerator * self.identity) // denominator

    def terms(self, x: int) -> list[int]:
        """
        Compute Padé-weighted power terms c_0 ^ x^0, c_1 * x^1, ..., c_order * x^order as
        fixed-point numbers.
        """
        # compute x^2, x^3, ..., x^order
        x_pow = x
        x_pows = [(x_pow := (x_pow * x) // self.identity) for _ in range(1, self.order)]
        # prepend x^1
        x_pows.insert(0, x)
        # multiply x^1, x^2, ..., x^(order-1) by Padé coefficients
        x_terms = [c * x_pow for (c, x_pow) in zip(self.coefficients[1:-1], x_pows[:-1])]
        # prepend constant c_0 * x^0 = c_0
        x_terms.insert(0, self.constant)
        # append c_order * x^order = x^order
        x_terms.append(x_pows[-1])
        return x_terms


def coefficients(order: int) -> list[int]:
    """Compute the Padé[N/N] coefficients of a given order for the exponential function."""
    return [math.factorial(2 * order - n) // (math.factorial(n) * math.factorial(order - n)) for n in range(order + 1)]


class BitShiftPadeApproximator(BitShiftApproximator):
    """Bit-shifted order-[N/N] Padé fixed-point approximator of the exponential function."""

    def __init__(self, decimals: int, order: int):
        super().__init__(decimals)
        self.remainder_approximator = PadeApproximator(decimals, order)
