
"""
Example: basic vector Zernike polynomial showcase
"""

import numpy as np
from typing import Optional

from physfields import ZernikeVector, VectorField

from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, Normalize

MAX_N = 4           # Maximum N
RESOLUTION = 20     # Resolution
SIZE = 12           # Plot size

plt.style.use('dark_background')


coeff: dict[(int, int, Optional[bool]), float] = {}


def main():
    x = np.linspace(-1, 1, 2 * RESOLUTION + 1)
    y = np.linspace(-1, 1, 2 * RESOLUTION + 1)
    xx, yy = np.meshgrid(x, y)

    for n in range(1, MAX_N):
        for l in range(-n, n + 1, 2):
            rs = [None] if abs(l) == n else [False, True]
            for r in rs:
                # Set coefficients randomly
                coeff[n, l, r] = np.random.normal(0, 0.5)**n

    for fno in range(0, 100):
        f = VectorField()

        for poly, c in coeff.items():
            n, l, r = poly
            f += c * ZernikeVector(n, l, r, masked=True)
            coeff[poly] += np.random.normal(0, 0.1)
            coeff[poly] *= 0.99

        f.mask = lambda x, y: (x**2 + y**2 >= 1)

        plt.grid(None)
        plt.gcf().set_size_inches(SIZE, SIZE)
        ax = plt.gca()
        ax.axis('off')
        ax.set_aspect('equal')
        f.plot_data(ax, xx, yy, colour='azimuth', scale=30)
        plt.tight_layout(pad=0)
        #plt.show()
        plt.savefig(f'{fno}.png')
        plt.close()
        print(fno)


if __name__ == "__main__":
    main()

