import mpmath

from expapprox.approximators.pade import PadeApproximator


class BitShiftPadeApproximator(PadeApproximator):
    """Order-[N/N] bit-shifted Padé fixed-point approximator of the exponential function."""

    __slots__ = ("log2", "log2half")

    def __init__(self, decimals: int, order: int):
        super().__init__(decimals, order)
        # precompute log(2) up to appropriate fixed-point precision
        with self.workdps:
            self.log2 = self.to_fixed(mpmath.log(2))
            self.log2half = self.log2 // 2

    def approx(self, x: int) -> int:
        # find integer quotient s.t. remainder is between -0.5*log(2) and 0.5*log(2)
        quotient = (x + self.log2half) // self.log2
        remainder = x - quotient * self.log2
        # compute exp(remainder) via Padé approximation
        preshifted = super().approx(remainder)
        # bitshift by quotient
        if x <= 0:
            # exp(remainder) // 2^quotient
            return preshifted >> abs(quotient)
        else:
            # exp(remainder) * 2^quotient
            return preshifted << quotient
