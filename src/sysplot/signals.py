import numpy as np

from .config import get_config


def heaviside(x: np.ndarray, default_value: float | None = None) -> np.ndarray:
    """Evaluate the Heaviside step function.

    Returns 0 for ``x < 0``, ``default_value`` for ``x == 0``, and 1 for
    ``x > 0``.

    Args:
        x: Input array.
        default_value: Value returned where ``x == 0``. Must be in the range [0, 1]. Defaults to
            :attr:`~sysplot.SysplotConfig.heaviside_default_value`, which is 1.

    Returns:
        Array of the same shape as ``x`` with Heaviside values.

    .. minigallery:: sysplot.heaviside
        :add-heading:
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"'x' must be a numpy ndarray, got {type(x).__name__!r}")
    default_value = (
        default_value
        if default_value is not None
        else get_config().heaviside_default_value
    )
    if not isinstance(default_value, (int, float)):
        raise TypeError(
            f"'default_value' must be a float, got {type(default_value).__name__!r}"
        )
    if not (0.0 <= default_value <= 1.0):
        raise ValueError(f"'default_value' must be in [0, 1], got {default_value!r}")
    return np.heaviside(x, default_value)
