import math

import matplotlib.pyplot as plt
from matplotlib import ticker

from expapprox.approximators import *
from expapprox.approximators.minimax import *
from expapprox.utils import float_range
from plots import rc_context, savefig

DECIMALS = 20


def relative_error_plot(cls, title: str):
    stepsize = 0.002
    xs = float_range(-5, 5, stepsize)
    xs = [x for x in xs if not (-stepsize < x < stepsize)]  # exclude e(0) = 1

    f, axs = plt.subplots(2, sharex=True, figsize=[7, 5])
    axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axs[0].set_xlim([xs[0], xs[-1]])
    axs[0].set_ylim([-20, 150])
    for ax in axs:
        ax.xaxis.set_tick_params(labelbottom=True)

    # titles and labels
    axs[0].set_title(title)
    axs[0].set_xlabel("$x$")
    axs[1].set_yscale("log")
    axs[1].set_title("Relative errors")
    axs[1].set_xlabel("$x$")

    for order in range(1, 4):
        approximator = cls(DECIMALS, order)
        # value plot
        axs[0].plot(xs, [approximator.try_call(x) for x in xs], color=f"C{order}", label=f"{order}")
        # relative error plot
        errs = approximator.benchmark(xs)
        axs[1].plot(xs, errs, color=f"C{order}")

    # reference value
    axs[0].plot(xs, [cls.ref(x) for x in xs], label="Reference", linestyle=":", color="C0", zorder=99)

    axs[0].legend(title="Order")

    return f


def pade_taylor_max_relative_errors_plot():
    stepsize = 0.05
    xs = float_range(-math.log(2) / 2, math.log(2) / 2, stepsize)
    xs = [x for x in xs if not (-stepsize < x < stepsize)]  # exclude e(0) = 1

    # compute max relative errors
    taylor_errs: dict[int, float] = {}
    pade_errs: dict[int, float] = {}
    for order in range(1, 6):
        taylor_errs[order] = max(TaylorApproximator(DECIMALS, order).benchmark(xs))
        pade_errs[order] = max(PadeApproximator(DECIMALS, order).benchmark(xs))

    f, ax = plt.subplots(1)

    # axes
    ax.set_yscale("log")
    ax.set_ylim(10**-16, 10**0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # titles and labels
    ax.set_title("Maximal relative errors")
    ax.set_xlabel("Order")

    # value plot
    ax.plot(taylor_errs.keys(), taylor_errs.values(), color="C0", label="Taylor")
    ax.plot(pade_errs.keys(), pade_errs.values(), color="C0", label="Padé", linestyle="--")

    ax.legend(title="Method", loc="lower left")

    return f


def bitshift_comparison_plot():
    stepsize = 0.002
    xs = float_range(-5, 5, stepsize)
    xs = [x for x in xs if not (-stepsize < x < stepsize)]  # exclude e(0) = 1

    f, axs = plt.subplots(3, 1, sharex=True, figsize=[7, 6])
    for ax in axs:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.set_xlim([xs[0], xs[-1]])

    for i, ax in enumerate(axs):
        order = i + 1

        # titles and labels
        ax.set_yscale("log")
        ax.set_xlabel("$x$")
        ax.set_title(f"Relative errors (order {order})")

        # relative errors
        ta = TaylorApproximator(DECIMALS, order)
        pa = PadeApproximator(DECIMALS, order)
        ba = BitShiftPadeApproximator(DECIMALS, order)
        ax.plot(xs, ta.benchmark(xs), color=f"C{order}", label=f"Taylor")
        ax.plot(xs, pa.benchmark(xs), color=f"C{order}", linestyle="--", label=f"Padé")
        ax.plot(xs, ba.benchmark(xs), color=f"C{order}", linestyle=":", label=f"Bit-shifted Padé")

    axs[0].legend(title="Method")

    return f


def minimax_comparison_plot():
    stepsize = 0.001
    xs = float_range(-math.log(2) / 2, math.log(2) / 2, stepsize)
    xs = [x for x in xs if not (-stepsize < x < stepsize)]  # exclude e(0) = 1

    f, axs = plt.subplots(3, 2, sharex=True, figsize=[7, 6])
    for ax in axs.flatten():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.set_xlim([xs[0], xs[-1]])

    # taylor vs. minimax polynomial
    for i, ax in enumerate(axs[:, 0]):
        order = i + 1
        # titles and labels
        ax.set_yscale("log")
        ax.set_xlabel("$x$")
        ax.set_title(f"Order {order}")
        # relative errors
        ta = TaylorApproximator(DECIMALS, order)
        ra = MinimaxPolynomialApproximator(order)
        ax.plot(xs, ta.benchmark(xs), color=f"C{order}", label=f"Taylor")
        ax.plot(xs, ra.benchmark(xs), color=f"C{order}", linestyle=":", label=f"Minimax poly")

    # pade vs. minimax rational
    for i, ax in enumerate(axs[:, 1]):
        order = i + 1
        # titles and labels
        ax.set_yscale("log")
        ax.set_xlabel("$x$")
        ax.set_title(f"Order {order}")
        # relative errors
        ta = PadeApproximator(DECIMALS, order)
        ra = MinimaxRationalApproximator(order)
        ax.plot(xs, ta.benchmark(xs), color=f"C{order}", label=f"Padé")
        ax.plot(xs, ra.benchmark(xs), color=f"C{order}", linestyle=":", label=f"Minimax ratio")

    f.suptitle("Relative errors")
    axs[0, 0].legend(title="Method", loc="lower right")
    axs[0, 1].legend(title="Method", loc="lower right")

    return f


def minimax_max_relative_errors_plot():
    stepsize = 0.05
    xs = float_range(-math.log(2) / 2, math.log(2) / 2, stepsize)
    xs = [x for x in xs if not (-stepsize < x < stepsize)]  # exclude e(0) = 1

    # compute max relative errors
    taylor_errs: dict[int, float] = {}
    pade_errs: dict[int, float] = {}
    minimax_poly_errs: dict[int, float] = {}
    minimax_rat_errs: dict[int, float] = {}

    import mpmath

    for order in range(1, 6):
        taylor_errs[order] = max(TaylorApproximator(DECIMALS, order).benchmark(xs))
        pade_errs[order] = max(PadeApproximator(DECIMALS, order).benchmark(xs))
        with mpmath.workdps(40):
            minimax_poly_errs[order] = max(MinimaxPolynomialApproximator(order).benchmark(xs))
            minimax_rat_errs[order] = max(MinimaxRationalApproximator(order).benchmark(xs))

    f, ax = plt.subplots(1)

    # axes
    ax.set_yscale("log")
    ax.set_ylim(10**-16, 10**0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # titles and labels
    ax.set_title("Maximal relative errors")
    ax.set_xlabel("Order")

    # value plot
    ax.plot(taylor_errs.keys(), taylor_errs.values(), color="C1", label="Taylor")
    ax.plot(pade_errs.keys(), pade_errs.values(), color="C2", label="Padé")
    ax.plot(minimax_poly_errs.keys(), minimax_poly_errs.values(), color="C1", label="Minimax poly", linestyle="--")
    ax.plot(minimax_rat_errs.keys(), minimax_rat_errs.values(), color="C2", label="Minimax ratio", linestyle="--")

    ax.legend(title="Method", loc="lower left")

    return f


def taylor_relative_errors():
    return relative_error_plot(TaylorApproximator, "Order-N Taylor approximation")


def pade_relative_errors():
    return relative_error_plot(PadeApproximator, "Order-[N/N] Padé approximation")


@rc_context
def main():
    savefig(taylor_relative_errors)
    savefig(pade_relative_errors)
    savefig(bitshift_comparison_plot)
    savefig(pade_taylor_max_relative_errors_plot)
    savefig(minimax_comparison_plot)
    savefig(minimax_max_relative_errors_plot)


if __name__ == "__main__":
    main()
