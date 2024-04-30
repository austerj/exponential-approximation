from dataclasses import dataclass

from expapprox.approximator import FixedPointApproximator


@dataclass(frozen=True, slots=True)
class MockApproximator(FixedPointApproximator):
    def __call__(self, x: int):
        return x

    def ref(self, x: int):
        return x


def test_identity():
    assert MockApproximator(3).identity == 1_000
    assert MockApproximator(4).identity == 10_000
    assert MockApproximator(5).identity == 100_000


def test_to_fixed_point():
    assert MockApproximator(3).to_fixed(1.02) == 1020
    assert MockApproximator(4).to_fixed(1) == 10000
    assert MockApproximator(5).to_fixed(59.201) == 5920100


def test_to_float():
    assert MockApproximator(3).to_float(1020) == 1.02
    assert MockApproximator(4).to_float(10000) == 1.0
    assert MockApproximator(5).to_float(5920100) == 59.201
