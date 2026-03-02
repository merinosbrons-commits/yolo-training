
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
    
    errors = positions - model.predict(time)
    sse = np.sum(errors**2)
    print(f"Sum of squared errors: {sse:.4f}")
    
    return velocity.tolist()
    



def constant_acceleration(positions: NDArray[np.float64]) -> list[float]:
    """Fits p(t)=b+v*t+0.5*a*t^2 and returns acceleration [ax, ay, az]."""

    n_samples = positions.shape[0]
    time = np.arange(1, n_samples + 1, dtype=np.float64)
    x = np.column_stack((time, 0.5 * (time ** 2))).astype(np.float64)

    model = LinearRegression()
    model.fit(x, positions)

    acceleration = model.coef_[:, 1]
  
    errors = positions - model.predict(x)
    sse = np.sum(errors**2)
    print(f"Sum of squared errors: {sse:.4f}")
    
    return acceleration.tolist()

if __name__ == "__main__":

    positions = np.array([
        [2.0, 0.0, 1.0],
        [1.08, 1.68, 2.38],
        [-0.83, 1.82, 2.49],
        [-1.97, 0.28, 2.15],
        [-1.31, -1.51, 2.59],
        [0.57, -1.91, 4.32]
    ], dtype=np.float64)

    v = constant_velocity(positions)
    a = constant_acceleration(positions)

    print("Velocity:", v)
    print("Acceleration:", a)