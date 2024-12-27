import numpy as np
from typing import Union

numeric = Union[int, float, np.number]
ArrayLike = Union[list[float], np.ndarray]
ArrayOrNumber = Union[numeric, ArrayLike]
