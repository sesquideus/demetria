import numpy as np

from typing import Optional

from ..core.scalarfield import ScalarField
from ..core.vectorfield import VectorField

from .scalar import Zernike, SR2, SROH



class ZernikeVector(VectorField):
    def __init__(self,
                 n: int,
                 l: int,
                 r: Optional[bool] = None,
                 *,
                 masked: bool = True):
        """
        Zernike vector basis field

        Implemented from Zhao & Burge (2008):
        Orthonormal vector polynomials in a unit circle,
        Part I: basis set derived from gradients of Zernike polynomials (2007), and
        Orthonormal vector polynomials in a unit circle,
        Part II : completing the basis set (2008).

        We introduce a simple notation using three indices n, l and r.

        n: int
            radial degree
        l: int
            azimuthal degree (-n <= l <= n, n = l (mod 2))
        r: bool or None
            rotational (True) or diverging (False) if |l| != n, else must be None (Laplacian)
        """
        if abs(l) > n or (l + n) % 2 != 0:
            raise ValueError("|l| must be <= n and also l = n (mod 2)")
        if abs(l) == n and r is not None:
            raise ValueError("Polynomials with |l| = n are always Laplacian")
        if abs(l) != n and r is None:
            raise ValueError("Polynomials with |l| != n must be rotational or diverging")

        self.n = n
        self.l = l
        self.r = r

        if masked:
            mask = lambda x, y: x**2 + y**2 >= 1
        else:
            mask = None

        if n == 0:
            super().__init__(ScalarField(), ScalarField(), mask=mask)
        elif n == 1:
            if l == -1:
                super().__init__(Zernike(0, 0), ScalarField(), mask=mask)
            else:
                super().__init__(ScalarField(), Zernike(0, 0), mask=mask)
        else:
            m = n - 1
            rot = -1 if r else 1

            if n == -l:
                super().__init__(SROH * Zernike(m, -m), SROH * Zernike(m, m), mask=mask)
            elif n == l:
                super().__init__(SROH * Zernike(m, m), -SROH * Zernike(m, -m), mask=mask)
            elif l == 0:
                super().__init__(SROH * Zernike(m, rot), (SROH * rot) * Zernike(m, -rot), mask=mask)
            elif abs(l) == 1:
                if l == -1:
                    super().__init__(0.5 * Zernike(m, -2),
                                     0.5 * ((rot * SR2) * Zernike(m, 0) - Zernike(m, 2)), mask=mask)
                else:
                    super().__init__(0.5 * (SR2 * Zernike(m, 0) + rot * Zernike(m, 2)),
                                     (0.5 * rot) * Zernike(m, -2), mask=mask)
            else:
                super().__init__(0.5 * (Zernike(m, l - 1) + rot * Zernike(m, l + 1)),
                                 0.5 * (rot * Zernike(m, -l - 1) - Zernike(m, -l + 1)), mask=mask)


    @classmethod
    def from_index(cls, index):
        pass

#    @staticmethod
