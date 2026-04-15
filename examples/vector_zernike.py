"""
Example: basic vector Zernike polynomial showcase
"""

import numpy as np

from physfields import ZernikeVector

from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, Normalize

MAX_N = 8           # Maximum N
RESOLUTION = 30     # Resolution
SIZE = 8            # Plot size

plt.style.use('dark_background')


def main():
    x = np.linspace(-1, 1, 2 * RESOLUTION + 1)
    y = np.linspace(-1, 1, 2 * RESOLUTION + 1)
    xx, yy = np.meshgrid(x, y)

    for n in range(1, MAX_N):
        for l in range(-n, n + 1, 2):
            rs = [None] if abs(l) == n else [False, True]
            for r in rs:
                if r is None:
                    dr = 1
                elif r == True:
                    dr = 2
                else:
                    dr = 0
                z = ZernikeVector(n, l, r, masked=True)

                ax = plt.subplot2grid((4 * MAX_N - 4, 2 * MAX_N), [4 * n + dr - 4, l + MAX_N - 1], 2, 2)
                ax.axis('off')
                ax.set_aspect('equal')

                z.plot_data(ax, xx, yy, colour='azimuth', scale=30)

    plt.grid(None)
    plt.gcf().set_size_inches(MAX_N * SIZE, 2 * MAX_N * SIZE)
    plt.tight_layout(pad=0)
    plt.savefig('out.pdf')
    plt.close()


if __name__ == "__main__":
    main()
