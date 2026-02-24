"""Example: plot_angle."""

import matplotlib.pyplot as plt
import sysplot as ssp


def main() -> None:
    ssp.apply_config()

    center = (0.0, 0.0)
    p1 = (1.0, 0.0)
    p2 = (0.6, 0.8)

    fig, ax = plt.subplots()
    ax.plot([center[0], p1[0]], [center[1], p1[1]])
    ax.plot([center[0], p2[0]], [center[1], p2[1]])

    ssp.plot_angle(center, p1, p2, text=r"$\theta$", ax=ax)

    ax.set_aspect("equal")
    ax.set_title("Angle annotation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()


if __name__ == "__main__":
    main()
