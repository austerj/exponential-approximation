import math


def int_range(start: int, end: int, step: int) -> list[int]:
    """Range of evenly-spaced integers."""
    length = math.floor((end - start) / step) + 1
    return [min(start + i * step, end) for i in range(length)]


def float_range(start: int | float, end: int | float, step: int | float) -> list[float]:
    """Range of evenly-spaced floats."""
    length = round((end - start) / step) + 1
    return [float(min(start + i * step, end)) for i in range(length)]


# https://www.math.utah.edu/~pa/math/e.html
_E = (
    "2"
    "71828"
    "18284"
    "59045"
    "23536"
    "02874"
    "71352"
    "66249"
    "77572"
    "47093"
    "69995"
    "95749"
    "66967"
    "62772"
    "40766"
    "30353"
    "54759"
    "45713"
    "82178"
    "52516"
    "64274"
    "27466"
    "39193"
    "20030"
    "59921"
    "81741"
    "35966"
    "29043"
    "57290"
    "03342"
    "95260"
    "59563"
    "07381"
    "32328"
    "62794"
    "34907"
    "63233"
    "82988"
    "07531"
    "95251"
    "01901"
    "15738"
    "34187"
    "93070"
    "21540"
    "89149"
    "93488"
    "41675"
    "09244"
    "76146"
    "06680"
    "82264"
    "80016"
    "84774"
    "11853"
    "74234"
    "54424"
    "37107"
    "53907"
    "77449"
    "92069"
    "55170"
    "27618"
    "38606"
    "26133"
    "13845"
    "83000"
    "75204"
    "49338"
    "26560"
    "29760"
    "67371"
    "13200"
    "70932"
    "87091"
    "27443"
    "74704"
)


def e(decimals: int) -> int:
    """Get fixed-point representation of e for specified number of decimals."""
    if decimals < 0:
        raise ValueError("Invalid decimals, must be greater than 0")
    if (missing_decimals := decimals - (len(_E) - 1)) > 0:
        return int(_E) * 10**missing_decimals
    return int(_E[: decimals + 1])
