import mpmath

from expapprox.approximators.pade import PadeApproximator


class BitShiftPadeApproximator(PadeApproximator):
    """Order-[N/N] bit-shifted Padé fixed-point approximator of the exponential function."""

    __slots__ = ("log2",)

    def __init__(self, decimals: int, order: int):
        super().__init__(decimals, order)
        # precompute log(2) up to appropriate fixed-point precision
        with self.workdps:
            self.log2 = self.to_fixed(mpmath.log(2))

    def approx(self, x: int) -> int:
        # quotient and remainder of floored division x // log(2)
        quotient, remainder = divmod(x, self.log2)
        # compute exp(remainder) via Padé approximation
        preshifted = super().approx(remainder)
        # bitshift by quotient
        if x <= 0:
            # exp(remainder) // 2^|quotient|
            return preshifted >> abs(quotient)
        else:
            # exp(remainder) * 2^quotient
            return preshifted << quotient