from abc import ABC
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray


MaskFunc: TypeAlias = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]


class Field(ABC):
    """
    Abstract base class for fields.
    """
    @staticmethod
    def UnitDiskMask(x, y):
        return x**2 + y**2 >= 1

    @staticmethod
    def UnitSquareMask(x, y):
        return np.abs(x) > 1 | np.abs(y) > 1

    def __init__(self,
                 *,
                 mask: MaskFunc = None,
                 name: str = ""):
        self.mask = mask
        self.name = name
        self.function: Callable = lambda x, y: 0

    def __call__(self, x, y):
        out = self.function(x, y)

        if self.mask is None:
            return out
        else:
            mask = self.mask(x, y)
            mask = np.stack([mask, mask])
            return np.ma.masked_where(mask, out)

