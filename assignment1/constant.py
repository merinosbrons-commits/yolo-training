import numpy as np
from numpy.typing import NDArray


def constant_velocity(positions: NDArray[np.float64]) -> list[float]:
    '''Calculates constant velocity, returning the points with that velocity.'''
    ...

def constant_accelaration(positions: NDArray[np.float64]) -> float:
    '''Calculates constant acceleration, returning the points with that velocity.'''
    ...