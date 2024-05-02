from abc import ABC


class ExponentialApproximationError(Exception, ABC):
    ...


# errors related to fixed-point numbers
class FixedPointError(ExponentialApproximationError):
    ...


class InvalidDecimalsError(FixedPointError):
    ...


# errors related to approximators
class ApproximatorError(ExponentialApproximationError):
    ...


class ApproximatorDomainError(ApproximatorError):
    ...


# errors related to integer tracking
class IntegerTrackerError(ExponentialApproximationError):
    ...
