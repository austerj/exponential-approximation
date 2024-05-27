import math

import matplotlib.pyplot as plt
import mpmath
import numpy as np
from matplotlib import ticker

from expapprox.approximators import BitShiftPadeApproximator
from expapprox.utils import float_range
from plots import rc_context, savefig

# parameters
C = 100_000
ALPHA = 0.02
RATE_ANNUAL = 0.05
BETA = 5

# configuration
ORDER = 3
DECIMALS = 16
PLOT_DECIMALS = 20

# computed constants
GAMMA: int = 60 * 60 * 24 * 365  # number of seconds in a year
RATE_CONTINUOUS: float = mpmath.log(1 + RATE_ANNUAL) / GAMMA
RELATIVE_ERROR_BOUND: float = ALPHA / (C * (1 + RATE_ANNUAL) * mpmath.exp(BETA))
X_UPPER: float = mpmath.log(1 + RATE_ANNUAL) * BETA

# bits required for input and approximation output
BAR_C = 10 ** (12 + 2)  # one trillion in cents
APPROX_UPPER: float = (1 + RATE_ANNUAL) * mpmath.exp(BETA) * 10**DECIMALS
BITS_DEPOSIT = math.ceil(math.log2(BAR_C))
BITS_APPROX = math.ceil(math.log2(APPROX_UPPER))
INTERMEDIARY_TEST_RANGE: list[float] = [*float_range(0, X_UPPER, 0.01), X_UPPER]
BITS_INTERMEDIARY = BitShiftPadeApproximator(DECIMALS, ORDER).max_bits(INTERMEDIARY_TEST_RANGE)
BITS_MULTIPLICATION = BITS_DEPOSIT + BITS_APPROX


def threshold_plot():
    stepsize = 0.01
    xs = float_range(0.02, 2 * X_UPPER, stepsize)

    f, ax = plt.subplots(1, 1, sharex=True)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlim([0, xs[-1]])

    # titles and labels
    ax.set_title(f"Relative error threshold")
    ax.set_yscale("log")
    ax.set_xlabel("$x$")

    for order in range(1, 5):
        # relative errors
        approximator = BitShiftPadeApproximator(PLOT_DECIMALS, order)
        ax.plot(xs, approximator.benchmark(xs), color=f"C{order}", label=order)

    ax.plot(
        [0, X_UPPER, X_UPPER],
        [RELATIVE_ERROR_BOUND, RELATIVE_ERROR_BOUND, 0],
        color="C6",
        linestyle=":",
        label=f"Threshold",
    )

    ax.legend(title="Order", loc="lower right")

    return f


def bits_plot():
    decimals = list(range(1, 26))
    approx_bits = [math.ceil(math.log2(10**d * (1 + RATE_ANNUAL) * mpmath.exp(BETA))) for d in decimals]

    f, axs = plt.subplots(2, 1, sharex=True, figsize=[7, 5])
    y_stepsize = 32
    y_upper = 5 * y_stepsize

    for ax in axs:
        ax.set_yticks(np.arange(0, y_upper + 1, y_stepsize))
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.set_xlim([decimals[0], decimals[-1]])
        ax.set_ylim([0, y_upper])
        ax.set_xlabel("$d_R$")

    # bits used for deposit and approximation product across number of digits
    axs[0].set_title(f"Required bits (multiplication)")
    axs[0].plot(decimals, [BITS_DEPOSIT + b for b in approx_bits], color=f"C0", label="Product")
    axs[0].plot(decimals, approx_bits, color=f"C1", label="Approximation")
    axs[0].plot([decimals[0], decimals[-1]], [BITS_DEPOSIT, BITS_DEPOSIT], color=f"C2", label=r"$\bar{C}$")
    axs[0].plot(
        [decimals[0], DECIMALS, DECIMALS], [BITS_MULTIPLICATION, BITS_MULTIPLICATION, 0], color=f"C3", linestyle=":"
    )

    # bits used for approximation across number of digits
    axs[1].set_title(f"Required bits (approximation)")
    approximator_bits = [BitShiftPadeApproximator(d, ORDER).max_bits(INTERMEDIARY_TEST_RANGE) for d in decimals]
    axs[1].plot(decimals, approximator_bits, color=f"C0", label="Intermediary values")
    axs[1].plot(decimals, approx_bits, color=f"C1", label="Approximation")
    axs[1].plot(
        [decimals[0], DECIMALS, DECIMALS], [BITS_INTERMEDIARY, BITS_INTERMEDIARY, 0], color=f"C3", linestyle=":"
    )

    axs[0].legend(loc="lower right")
    axs[1].legend(loc="lower right")

    return f


@rc_context
def main():
    savefig(threshold_plot)
    savefig(bits_plot)


if __name__ == "__main__":
    main()
