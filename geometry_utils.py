import numpy as np

def skew(omega: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3D vector.

    Parameters
    - omega: length-3 array-like representing angular velocity

    Returns
    - 3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0],
    ])


