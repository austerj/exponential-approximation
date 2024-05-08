import matplotlib.pyplot as plt
from matplotlib import ticker

from expapprox.approximators import *
from expapprox.utils import float_range
from plots import rc_context, savefig


def value_plot(cls, title: str, is_pade: bool = False):
    start, end = -4, 4
    xs = float_range(start, end, 0.01)

    f, ax = plt.subplots(1)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylim([-10, 60])

    # titles and labels
    ax.set_title(title)
    ax.set_xlabel("$x$")

    for order in range(1, 4):
        approximator = cls(10, order)
        # value plot
        ax.plot(
            xs,
            [approximator.try_call(x) for x in xs],
            color=f"C{order}",
            label=f"[{order}/{order}]" if is_pade else f"{order}",
        )

    # reference value
    ax.plot(xs, [cls.ref(x) for x in xs], label="Reference", linestyle=":", color="C0", zorder=99)

    ax.legend(title="Order")

    return f


def taylor_values():
    return value_plot(TaylorApproximator, "Order-N Taylor approximation")


def pade_values():
    return value_plot(PadeApproximator, "Order-[N/N] Padé approximation", True)


def bshift_pade_values():
    return value_plot(BitShiftPadeApproximator, "Order-[N/N] bit-shifted Padé approximation", True)


@rc_context
def main():
    savefig(taylor_values)
    savefig(pade_values)
    savefig(bshift_pade_values)


if __name__ == "__main__":
    main()
