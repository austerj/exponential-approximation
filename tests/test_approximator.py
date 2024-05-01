from expapprox.approximator import FixedPointApproximator


class MockApproximator(FixedPointApproximator):
    def approx(self, x: int) -> int:
        return x

    @classmethod
    def ref(cls, x: float) -> float:
        return 1.0


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


def test_benchmark():
    assert MockApproximator(3).benchmark([1.0, 1.5, 2.0, 2.5]) == [0.0, 0.5, 1.0, 1.5]
