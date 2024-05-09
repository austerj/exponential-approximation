import matplotlib.pyplot as plt
from matplotlib import ticker

from expapprox.approximators import *
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


def comparison_plot():
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
        axs[i].plot(xs, ta.benchmark(xs), color=f"C{order}", label=f"Taylor")
        axs[i].plot(xs, pa.benchmark(xs), color=f"C{order}", linestyle="--", label=f"Padé")
        axs[i].plot(xs, ba.benchmark(xs), color=f"C{order}", linestyle=":", label=f"Bit-shifted Padé")

    # reference value
    axs[0].legend(title="Method")

    return f


def taylor_relative_errors():
    return relative_error_plot(TaylorApproximator, "Order-N Taylor approximation")


def pade_relative_errors():
    return relative_error_plot(PadeApproximator, "Order-[N/N] Padé approximation")


def bshift_pade_relative_errors():
    return relative_error_plot(BitShiftPadeApproximator, "Order-[N/N] bit-shifted Padé approximation")


@rc_context
def main():
    savefig(taylor_relative_errors)
    savefig(pade_relative_errors)
    savefig(bshift_pade_relative_errors)
    savefig(comparison_plot)


if __name__ == "__main__":
    main()
