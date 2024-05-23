from abc import ABC

import mpmath

from expapprox.approximator import FixedPointExponentialApproximator


class BitShiftApproximator(FixedPointExponentialApproximator, ABC):
    """Base class for bit-shifted fixed-point approximator of the exponential function."""

    __slots__ = ("remainder_approximator", "log2", "log2half")
    remainder_approximator: FixedPointExponentialApproximator

    def __init__(self, decimals: int):
        super().__init__(decimals)
        # precompute log(2) up to appropriate fixed-point precision
        with self.workdps:
            self.log2 = self.to_fixed(mpmath.log(2))
            self.log2half = self.log2 // 2

    def _fields(self):
        return self.remainder_approximator._fields()

    def approx(self, x: int) -> int:
        # find integer quotient s.t. remainder is between -0.5*log(2) and 0.5*log(2)
        quotient = (x + self.log2half) // self.log2
        remainder = x - quotient * self.log2
        # compute exp(remainder) via remainder approximator
        preshifted = self.remainder_approximator.approx(remainder)
        # bitshift by quotient
        if x <= 0:
            # exp(remainder) // 2^quotient
            return preshifted >> abs(quotient)
        else:
            # exp(remainder) * 2^quotient
            return preshifted << quotient
