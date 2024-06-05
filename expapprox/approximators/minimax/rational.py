import mpmath

from expapprox import errors
from expapprox.approximator import ExponentialApproximator

# minimax rational coefficients computed via MATLAB chebfun package (see coeff.m)
MINIMAX_RATIONAL = {
    1: (
        (
            "1.0000000000000000",
            "1.9951230090905039",
        ),
        (
            "-0.9802384712903994",
            "1.9945394068684223",
        ),
    ),
    2: (
        (
            "1.0000000000000000",
            "5.9910080216065911",
            "11.9579893099513441",
        ),
        (
            "0.9880690225786635",
            "-5.9669083099599627",
            "11.9579875493587569",
        ),
    ),
    3: (
        (
            "1.0000000000000000",
            "11.9892808348301951",
            "59.8842261733986732",
            "119.6656635650113572",
        ),
        (
            "-0.9914599095026225",
            "11.9378869080663428",
            "-59.7814371744344228",
            "119.6656635612295787",
        ),
    ),
    4: (
        (
            "1.0000000000000000",
            "19.9842829398787600",
            "179.7188587998745390",
            "838.1320185010351906",
            "1675.3400898161289660",
        ),
        (
            "0.9923137290653044",
            "-19.8919149669371222",
            "179.2568852086743334",
            "-837.2080713146019662",
            "1675.3400898161228270",
        ),
    ),
    5: (
        (
            "1.0000000000000000",
            "21.9446091521355022",
            "225.9576129377199152",
            "1317.5889015366262811",
            "4208.3110461309424863",
            "5657.0517857221739177",
        ),
        (
            "-0.0000000078664024",
            "0.6402427464310314",
            "-10.8932364260787988",
            "69.6822701690608142",
            "-62.1962517309060701",
            "-1448.7407395911168351",
            "5657.0517857221666418",
        ),
    ),
    6: (
        (
            "1.0000000000000000",
            "20.1240535434826846",
            "183.4367857794373151",
            "878.9791794821758231",
            "1889.3021785500918668",
            "384.8483752752718487",
            "-485.3990833051483946",
        ),
        (
            "0.9374980053482066",
            "-18.6264976825290418",
            "164.7423379044435592",
            "-736.9989642115033348",
            "1261.7542616220155196",
            "870.2474585804197886",
            "-485.3990833051479967",
        ),
    ),
}


class MinimaxRationalApproximator(ExponentialApproximator):
    """Order-N minimax rational approximator of the exponential function on [-log(2)/2, log(2)/2]."""

    __slots__ = ("order", "p_coefficients", "q_coefficients")

    def __call__(self, x: int | float) -> float:
        p = mpmath.polyval(self.p_coefficients, x)
        q = mpmath.polyval(self.q_coefficients, x)
        return p / q

    def __init__(self, order: int):
        if order not in MINIMAX_RATIONAL:
            raise errors.ApproximatorError(f"Invalid order {order}; no minimax coefficients available")
        self.order = order
        self.p_coefficients = [mpmath.mpf(x) for x in MINIMAX_RATIONAL[order][0]]
        self.q_coefficients = [mpmath.mpf(x) for x in MINIMAX_RATIONAL[order][1]]

    def _fields(self):
        return [*super()._fields(), f"order={self.order}"]
