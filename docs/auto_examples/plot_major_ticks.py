"""
Example: set_major_tick_labels
"""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp


def main() -> None:
    ssp.apply_config()

    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ssp.set_major_tick_labels(
        label=r"\pi",
        unit=np.pi,
        numerator=1,
        denominator=2,
        axis=ax.xaxis,
    )

    ax.set_title("Sine with pi ticks")
    ax.set_xlabel(r"$t$ [rad]")
    ax.set_ylabel("amplitude")

    plt.show()


if __name__ == "__main__":
    main()
