# Abstract class for estimator

from abc import ABC
from abc import abstractmethod
import numpy as np

from ml_from_scratch.base.types import ArrayLike, ArrayOrNumber


class Estimator(ABC):

    @property
    @abstractmethod
    def weights(self):
        pass

    @weights.setter
    @abstractmethod
    def weights(self, value: ArrayLike) -> None:
        pass

    def __init__(self):
        if not any(hasattr(self, attr) for attr in ['weights', 'coef', 'feature_importances']):
            raise NotImplementedError(
                "Estimator must implement at least one of the attributes: 'weights', 'coef', or 'feature_importances'."
            )

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
        pass

    @abstractmethod
    def predict(self, X: ArrayOrNumber) -> np.array:
        pass
