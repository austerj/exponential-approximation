import inspect
import os
import typing

import matplotlib as mpl
from cycler import cycler
from matplotlib.figure import Figure

rc_context = mpl.rc_context(
    {
        # figure
        "figure.figsize": [7, 3],
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.08334,
        "figure.constrained_layout.w_pad": 0.08334,
        # font
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 11,
        "font.size": 11,
        # plot styling
        # https://scottplot.net/cookbook/4.1/colors/#colorblind-friendly
        "axes.prop_cycle": cycler(
            color=[
                "#000000",
                "#E69F00",
                "#56B4E9",
                "#009E73",
                "#F0E442",
                "#0072B2",
                "#D55E00",
                "#CC79A7",
            ]
        ),
        "lines.color": "black",
        "lines.linewidth": 2,
        # legend
        "legend.fontsize": "small",
        # axes
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # output
        "savefig.format": "pdf",
    }
)


def savefig(f: typing.Callable[[], Figure] | Figure, fname: typing.Optional[str] = None, *args, **kwargs):
    # get path to .out dir
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".out")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # make figure if received callable
    fig = f if isinstance(f, Figure) else f()
    # default to function name, else filename (ignoring ".py") of calling module
    default_fname = os.path.basename(inspect.stack()[1].filename[:-3]) if isinstance(f, Figure) else f.__name__
    # save to .out/{filename}.pdf
    fig.savefig(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            ".out",
            default_fname if fname is None else fname,
        ),
        *args,
        **kwargs,
    )
