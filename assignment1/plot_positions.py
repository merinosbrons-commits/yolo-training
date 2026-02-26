import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

def plot_positions(positions: NDArray[np.float64]) -> None:
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='red', marker='o', label='Positions')
    ax.plot(x, y, z, label='Route', linestyle='--', alpha=0.6)

    [ax.text(x[i] + 0.1, y[i], z[i], f't = {i+1}', size=10, zorder=1, color='black') for i in range(len(positions))]

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Route Visualisation of the drone')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    positions = np.array([
        [2.0, 0.0, 1.0],
        [1.08, 1.68, 2.38],
        [-0.83, 1.82, 2.49],
        [-1.97, 0.28, 2.15],
        [-1.31, -1.51, 2.59],
        [0.57, -1.91, 4.32]
    ], dtype=np.float64)

    plot_positions(positions)