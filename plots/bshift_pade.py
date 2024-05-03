import matplotlib.pyplot as plt
from matplotlib import ticker

from expapprox.approximators.bshift_pade import BitShiftPadeApproximator
from expapprox.utils import float_range
from plots import rc_context, savefig


def bshift_pade_plot():
    xs = float_range(-1.5, 1.5, 0.1)

    f, axs = plt.subplots(2, sharex=True, figsize=[7, 5])
    axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axs[0].set_xlim([xs[0], xs[-1]])
    for ax in axs:
        ax.xaxis.set_tick_params(labelbottom=True)

    # titles and labels
    axs[0].set_title("Order-[N/N] bit-shifted Padé approximation")
    axs[0].set_xlabel("$x$")
    axs[1].set_title("Relative errors")
    axs[1].set_xlabel("$x$")

    for order in range(1, 5):
        approximator = BitShiftPadeApproximator(10, order)
        # value plot
        axs[0].plot(xs, [approximator(x) for x in xs], color=f"C{order}", label=f"[{order}/{order}]")
        # relative error plot
        errs = approximator.benchmark(xs)
        axs[1].plot(xs, errs, color=f"C{order}")

    # reference value
    axs[0].plot(
        xs, [BitShiftPadeApproximator.ref(x) for x in xs], label="Reference", linestyle=":", color="C0", zorder=99
    )

    axs[0].legend(title="Order")

    return f


def bshift_pade_plot_wide():
    xs = float_range(-4, 4, 0.05)

    f, ax = plt.subplots(1)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlim([xs[0], xs[-1]])

    # titles and labels
    ax.set_title("Order-[N/N] bit-shifted Padé approximation")
    ax.set_xlabel("$x$")

    for order in range(1, 5):
        approximator = BitShiftPadeApproximator(10, order)
        # value plot
        ax.plot(xs, [approximator.try_call(x) for x in xs], color=f"C{order}", label=f"[{order}/{order}]")

    # reference value
    ax.plot(xs, [BitShiftPadeApproximator.ref(x) for x in xs], label="Reference", linestyle=":", color="C0", zorder=99)

    ax.legend(title="Order")

    return f


@rc_context
def main():
    savefig(bshift_pade_plot)
    savefig(bshift_pade_plot_wide)


if __name__ == "__main__":
    main()
