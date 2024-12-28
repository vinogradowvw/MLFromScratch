from ml_from_scratch.base.types import ArrayLike, numeric
import numpy as np


def log_loss(y_true, y_pred_proba):
    """
    Calculates log-loss value for one prediction
    """
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
