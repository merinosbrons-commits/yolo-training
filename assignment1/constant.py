import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression


def constant_velocity(positions: NDArray[np.float64]) -> list[float]:
    """Fits p(t)=b+v*t and returns velocity [vx, vy, vz]."""

    n_samples = positions.shape[0]
    time = np.arange(1, n_samples + 1, dtype=np.float64).reshape(-1, 1)

    model = LinearRegression()
    model.fit(time, positions)

    velocity = model.coef_[:, 0]
    return velocity.tolist()


def constant_accelaration(positions: NDArray[np.float64]) -> list[float]:
    """Fits p(t)=b+v*t+0.5*a*t^2 and returns acceleration [ax, ay, az]."""

    n_samples = positions.shape[0]
    time = np.arange(1, n_samples + 1, dtype=np.float64)
    x = np.column_stack((time, 0.5 * (time ** 2))).astype(np.float64)

    model = LinearRegression()
    model.fit(x, positions)

    acceleration = model.coef_[:, 1]
    return acceleration.tolist()