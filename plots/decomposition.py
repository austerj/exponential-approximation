import math

import matplotlib.pyplot as plt
from matplotlib import ticker

from expapprox.utils import float_range
from plots import rc_context, savefig


def decomposition_plot():
    stepsize = 0.002
    xs = float_range(-5, 5, stepsize)
    ks = [math.floor(x / math.log(2) + 0.5) for x in xs]
    rs = [x - k * math.log(2) for x, k in zip(xs, ks)]

    # values
    exp_xs = [math.exp(x) for x in xs]
    exp_rs = [math.exp(r) for r in rs]
    k_pows = [2**k for k in ks]

    f, axs = plt.subplots(3, sharex=True, figsize=[7, 6])
    # xaxis
    axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    for ax in axs:
        ax.set_xlim([xs[0], xs[-1]])
        ax.set_xlabel("$x$")
        ax.xaxis.set_tick_params(labelbottom=True)
    # ylims
    axs[0].set_ylim([-20, 150])
    axs[1].set_ylim([0.5, 1.5])
    axs[2].set_ylim([2**-8, 2**8])

    axs[0].set_title("Exponential function decomposition")

    # decomposition plot
    axs[0].plot(xs, exp_xs, color=f"C0", label="$e^x$")
    axs[0].plot(xs, k_pows, color=f"C1", label="$2^k$")
    axs[0].plot([], [], color=f"C2", label=r"$e^{x - k \log 2}$")  # shared legend on first subplot

    axs[1].plot(xs, exp_rs, color=f"C2")

    axs[2].plot(xs, exp_xs, color=f"C0")
    axs[2].plot(xs, k_pows, color=f"C1")
    axs[2].plot(xs, exp_rs, color=f"C2")

    axs[2].set_yscale("log", base=2)
    axs[2].yaxis.set_major_locator(ticker.LogLocator(base=2, numticks=5))

    axs[0].legend()

    return f


@rc_context
def main():
    savefig(decomposition_plot)


if __name__ == "__main__":
    main()
