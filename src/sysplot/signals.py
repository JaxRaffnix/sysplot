import numpy as np

def heaviside(x: np.ndarray, default_value: float = 1) -> np.ndarray:
    """Heaviside step function with default value for x=0."""
    return np.heaviside(x, default_value)