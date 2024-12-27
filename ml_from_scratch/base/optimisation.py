from typing import Callable
from typing import Optional
import numpy as np

from ml_from_scratch.base.types import ArrayLike, ArrayOrNumber, numeric
from ml_from_scratch.base.Estimator import Estimator


def compute_gradient(loss_function: Callable[[numeric, numeric], numeric],
                     estimator: Estimator,
                     weights: np.ndarray,
                     x: numeric,
                     y: numeric,
                     epsilon=1e-5
                     ) -> np.ndarray:
    """
    Computes gradient of a loss function in point (prediction(x), y).
    """
    grad = np.zeros_like(weights)

    for i in range(len(weights)):
        weights_plus = weights.copy()
        weights_minus = weights.copy()

        weights_plus[i] += epsilon
        weights_minus[i] -= epsilon

        estimator_minus = estimator
        estimator_plus = estimator

        estimator_minus.weights = weights_minus
        estimator_plus.weights = weights_plus

        loss_plus = loss_function(y, estimator_plus.predict(x)[0])
        loss_minus = loss_function(y, estimator_plus.predict(x)[0])

        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return grad


def has_converged(R_emp_current: numeric,
                  R_emp_previous: numeric,
                  w_current: np.ndarray,
                  w_previous: np.ndarray,
                  tol=1e-6):
    R_emp_converged = abs(R_emp_current - R_emp_previous) < tol
    w_converged = np.linalg.norm(w_current - w_previous) < tol
    return R_emp_converged and w_converged


def sgd(X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        estimator: Estimator,
        loss_function: Callable[[numeric, numeric], numeric],
        learning_rate: numeric,
        regularization: Optional[Callable[[numeric], numeric]],
        lambda_: numeric = 0.01,
        ) -> np.ndarray:

    emp_r = 0
    emp_r_current = 1
    weights = np.array(weights)
    X, y = np.ndarray(X), np.ndarray(y)
    weights_old = np.array(weights) + 1

    while has_converged(emp_r, emp_r_current, weights, weights_old):
        emp_r = emp_r_current
        weights_old = weights.copy()

        i = np.random.choice(len(X))
        x_i, y_i = X[i], y[i]
        X, y = np.delete(X, i), np.delete(y, i)

        pred = estimator.predict(x_i)
        err = loss_function(y_i, pred)

        grad = compute_gradient(loss_function, estimator, weights, x_i, y_i)

        weights = weights - learning_rate * grad

        emp_r_current = lambda_*err + (1 - lambda_) * emp_r

    return weights
