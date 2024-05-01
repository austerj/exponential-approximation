import pytest

from expapprox.tracker import IntegerTracker


@pytest.fixture
def tracker():
    return IntegerTracker()


def test_context():
    with IntegerTracker() as tracker:
        x = tracker.int(10)
        x += 15
        x -= 5
        assert tracker.min_int == 5
        assert tracker.max_int == 25
        assert not tracker.signed


def test_signed(tracker: IntegerTracker):
    x = tracker.int(0)
    assert not tracker.signed
    x -= 1
    assert tracker.signed


def test_bits(tracker: IntegerTracker):
    # 0 bits for empty range
    assert tracker.bits == 0
    # 0 bits for { 0 }
    x = tracker.int(0)
    assert tracker.bits == 0
    # 1 bit for 0 to 1
    x += 1
    assert tracker.bits == 1
    # 2 bits for 0 to 3
    x += 2
    assert tracker.bits == 2
    # 3 bits for 0 to 4
    x += 1
    assert tracker.bits == 3
    # 3 bits for 0 to 7
    x += 3
    assert tracker.bits == 3
    # 4 bits for 0 to 8
    x += 1
    assert tracker.bits == 4
    # 5 bits for -1 to 8
    assert not tracker.signed
    y = tracker.int(-1)
    assert tracker.bits == 5
    assert tracker.signed


def test_propagation(tracker: IntegerTracker):
    c0 = 9
    c1 = 2
    c2 = 13

    def f(x: int) -> int:
        return (x + c0) // ((x - c1) // c1 - c2) - c1 * x

    running_min, running_max = 0, 0
    # starting from smaller absolute values to ensure tracked min / max is changing frequently
    for x in [5, -2, -3, 12, -9, 22, -59, 13, 49, 123, -5012, 8501]:
        # compute intermediary values from untracked ints
        y0 = x + c0
        y1 = x - c1
        y2 = y1 // c1
        y3 = y2 - c2
        y4 = y0 // y3
        y5 = c1 * x
        y6 = y4 - y5
        running_min = min(running_min, x, y0, y1, y2, y3, y4, y5, y6, c0, c1, c2)
        running_max = max(running_max, x, y0, y1, y2, y3, y4, y5, y6, c0, c1, c2)
        # run function with tracked int
        f(tracker.int(x))
        # check that tracker aligns with "manually" computed intermediary values
        assert tracker.min_int == running_min
        assert tracker.max_int == running_max
