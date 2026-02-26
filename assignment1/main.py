import numpy as np

from plot_positions import plot_positions


def main():
    positions = np.array([
        [2.0, 0.0, 1.0],
        [1.08, 1.68, 2.38],
        [-0.83, 1.82, 2.49],
        [-1.97, 0.28, 2.15],
        [-1.31, -1.51, 2.59],
        [0.57, -1.91, 4.32]
    ], dtype=np.float64)

    plot_positions(positions)


if __name__ == "__main__":    
    main()