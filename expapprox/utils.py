import math


def int_range(start: int, end: int, step: int) -> list[int]:
    """Range of evenly-spaced integers."""
    length = math.floor((end - start) / step) + 1
    return [min(start + i * step, end) for i in range(length)]


def float_range(start: int | float, end: int | float, step: int | float) -> list[float]:
    """Range of evenly-spaced floats."""
    length = round((end - start) / step) + 1
    return [float(min(start + i * step, end)) for i in range(length)]
