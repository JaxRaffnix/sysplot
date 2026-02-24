"""Example: set_minor_log_ticks."""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp


def main() -> None:
    ssp.apply_config()

    x = np.logspace(0, 3, 300)
    y = 1 / x

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.plot(x, y)

    ssp.set_minor_log_ticks(axis=ax.xaxis)

    ax.set_title("Log axis with minor ticks")
    ax.set_xlabel("frequency")
    ax.set_ylabel("magnitude")

    plt.show()

