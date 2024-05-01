from __future__ import annotations

import collections.abc
import typing
from abc import ABC
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

R = typing.TypeVar("R")


@dataclass
class IntegerTracker(AbstractContextManager):
    """Context manager for tracking behavior of integers during intermediary operations."""

    # tracked attributes
    min_int: int | None = field(default=None, init=False)
    max_int: int | None = field(default=None, init=False)
    int: typing.Type[_TrackedInteger] = field(init=False)

    def __post_init__(self):
        # instance-specific TrackedInteger subclass
        class TrackedInteger(_TrackedInteger):
            tracker = self

        self.int = TrackedInteger

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: typing.Type | None,
    ) -> bool | None:
        pass

    @property
    def signed(self):
        """Flag denoting if signed integers have been used."""
        return self.min_int is not None and self.min_int < 0

    @property
    def bits(self):
        """Number of bits required to represent all tracked integers (including sign)."""
        min_bit_length = self.min_int.bit_length() if self.min_int is not None else 0
        max_bit_length = self.max_int.bit_length() if self.max_int is not None else 0
        return max(min_bit_length, max_bit_length) + int(self.signed)

    def register(self, value: int):
        """Register integer with tracker context."""
        if not isinstance(value, int):
            raise TypeError(f"Cannot track non-integer value: {value}")
        # update min / max values
        self.min_int = value if self.min_int is None else min(self.min_int, value)
        self.max_int = value if self.max_int is None else max(self.max_int, value)


# note: "lying" in the return type here to preserve (int) type hints of wrapped methods
def tracked(fn: typing.Callable[..., R]) -> typing.Callable[..., R]:
    """Register new value with IntegerTracker and return value as TrackedInteger."""

    def inner(tracked_int: _TrackedInteger, *args):
        # register all int args
        for arg in args:
            if isinstance(arg, int):
                tracked_int.tracker.register(arg)
        # apply operation
        value = fn(tracked_int, *args)
        if isinstance(value, collections.abc.Sequence):
            # register elements of returned sequence
            for v in value:
                tracked_int.tracker.register(v)
            return tuple(tracked_int.__class__(v) for v in value)
        else:
            # register return value
            tracked_int.tracker.register(typing.cast(int, value))
            return tracked_int.__class__(typing.cast(int, value))

    return inner  # type: ignore


class _TrackedInteger(int, ABC):
    """
    Subclass of built-in int that registers with IntegerTracker on every operation involving itself
    and returns a new tracked integer.
    """

    tracker: IntegerTracker

    def __new__(cls, value: int):
        cls.tracker.register(value)
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    # track all relevant int dunder methods
    @tracked
    def __add__(self, value: int, /) -> int:
        return super().__add__(value)

    @tracked
    def __sub__(self, value: int, /) -> int:
        return super().__sub__(value)

    @tracked
    def __mul__(self, value: int, /) -> int:
        return super().__mul__(value)

    @tracked
    def __floordiv__(self, value: int, /) -> int:
        return super().__floordiv__(value)

    @tracked
    def __truediv__(self, value: int, /) -> float:
        return super().__truediv__(value)

    @tracked
    def __mod__(self, value: int, /) -> int:
        return super().__mod__(value)

    @tracked
    def __divmod__(self, value: int, /) -> tuple[int, int]:
        return super().__divmod__(value)

    @tracked
    def __radd__(self, value: int, /) -> int:
        return super().__radd__(value)

    @tracked
    def __rsub__(self, value: int, /) -> int:
        return super().__rsub__(value)

    @tracked
    def __rmul__(self, value: int, /) -> int:
        return super().__rmul__(value)

    @tracked
    def __rfloordiv__(self, value: int, /) -> int:
        return super().__rfloordiv__(value)

    @tracked
    def __rtruediv__(self, value: int, /) -> float:
        return super().__rtruediv__(value)

    @tracked
    def __rmod__(self, value: int, /) -> int:
        return super().__rmod__(value)

    @tracked
    def __rdivmod__(self, value: int) -> tuple[int, int]:
        return super().__rdivmod__(value)

    @tracked
    def __pow__(self, value: int, /) -> int:
        return super().__pow__(value)

    @tracked
    def __rpow__(self, value: int, /) -> int:
        return super().__rpow__(value)

    @tracked
    def __and__(self, value: int, /) -> int:
        return super().__and__(value)

    @tracked
    def __or__(self, value: int, /) -> int:
        return super().__or__(value)

    @tracked
    def __xor__(self, value: int, /) -> int:
        return super().__xor__(value)

    @tracked
    def __lshift__(self, value: int, /) -> int:
        return super().__lshift__(value)

    @tracked
    def __rshift__(self, value: int, /) -> int:
        return super().__rshift__(value)

    @tracked
    def __rand__(self, value: int, /) -> int:
        return super().__rand__(value)

    @tracked
    def __ror__(self, value: int, /) -> int:
        return super().__ror__(value)

    @tracked
    def __rxor__(self, value: int, /) -> int:
        return super().__rxor__(value)

    @tracked
    def __rlshift__(self, value: int, /) -> int:
        return super().__rlshift__(value)

    @tracked
    def __rrshift__(self, value: int, /) -> int:
        return super().__rrshift__(value)

    @tracked
    def __neg__(self) -> int:
        return super().__neg__()

    @tracked
    def __pos__(self) -> int:
        return super().__pos__()

    @tracked
    def __invert__(self) -> int:
        return super().__invert__()

    @tracked
    def __trunc__(self) -> int:
        return super().__trunc__()

    @tracked
    def __ceil__(self) -> int:
        return super().__ceil__()

    @tracked
    def __floor__(self) -> int:
        return super().__floor__()

    @tracked
    def __round__(self, ndigits: typing.SupportsIndex) -> int:
        return super().__round__(ndigits)

    @tracked
    def __int__(self) -> int:
        return super().__int__()

    @tracked
    def __abs__(self) -> int:
        return super().__abs__()
