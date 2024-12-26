# Abstract class for estimator

from abc import ABC
from abc import abstractmethod
from typing import Union
import pandas as pd
import numpy as np

ArrayLike = Union[list, np.ndarray, pd.DataFrame, pd.Series]


class Estimator(ABC):

    def __init__(self):
        if not any(hasattr(self, attr) for attr in ['weights', 'coef', 'feature_importances']):
            raise NotImplementedError(
                "Estimator must implement at least one of the attributes: 'weights', 'coef', or 'feature_importances'."
            )

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
        pass

    @abstractmethod
    def predict(self, X: ArrayLike):
        pass
