"""
Example: basic scalar Zernike polynomials
"""

import numpy as np

from physfields import Zernike

from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm

# Maximum N
MaxN = 8

# Resolution
Res = 200

def main():
    for n in range(0, MaxN):
        for l in range(-n, n + 1, 2):
            z = Zernike(n, l, masked=True)

            x = np.linspace(-1, 1, 2 * Res + 1)
            y = np.linspace(-1, 1, 2 * Res + 1)
            xx, yy = np.meshgrid(x, y)

            ax = plt.subplot2grid((2 * MaxN, 2 * MaxN), [2 * n, l + MaxN - 1], 2, 2)
            ax.axis('off')
            plt.imshow(z(xx, yy), cmap='bwr', norm=CenteredNorm(0))

    plt.grid(None)
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    main()

