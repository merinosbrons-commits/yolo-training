
import numpy as np
from numpy.typing import NDArray


def _validate_positions(positions: NDArray[np.float64], minimum_points: int) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if positions.shape[0] < minimum_points:
        raise ValueError(f"positions must contain at least {minimum_points} points")


def _gradient_descent(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    learning_rate: float,
    iterations: int,
    tolerance: float,
) -> NDArray[np.float64]:
    n_samples, n_features = x.shape
    n_targets = y.shape[1]

    weights = np.zeros((n_features, n_targets), dtype=np.float64)
    previous_loss = np.inf

    for _ in range(iterations):
        predictions = x @ weights
        errors = predictions - y
        loss = np.mean(errors ** 2)

        gradient = (2.0 / n_samples) * (x.T @ errors)
        weights -= learning_rate * gradient

        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss

    return weights


def constant_velocity(positions: NDArray[np.float64]) -> list[float]:
    """Fits p(t)=b+v*t and returns velocity [vx, vy, vz]."""
    _validate_positions(positions, minimum_points=2)

    n_samples = positions.shape[0]
    time = np.arange(1, n_samples + 1, dtype=np.float64)
    x = np.column_stack((np.ones_like(time), time)).astype(np.float64)

    weights = _gradient_descent(
        x,
        positions,
        learning_rate=1e-2,
        iterations=100_000,
        tolerance=1e-12,
    )

    velocity = weights[1, :]
    return velocity.tolist()
    

def constant_acceleration(positions: NDArray[np.float64]) -> list[float]:
    """Fits p(t)=b+v*t+0.5*a*t^2 and returns acceleration [ax, ay, az]."""
    _validate_positions(positions, minimum_points=3)

    n_samples = positions.shape[0]
    time = np.arange(1, n_samples + 1, dtype=np.float64)
    x = np.column_stack((time, 0.5 * (time ** 2))).astype(np.float64)

    weights = _gradient_descent(
        x,
        positions,
        learning_rate=5e-4,
        iterations=200_000,
        tolerance=1e-12,
    )

    acceleration = weights[1, :]
    return acceleration.tolist()


def constant_accelaration(positions: NDArray[np.float64]) -> list[float]:
    return constant_acceleration(positions)

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