import numpy as np


class KalmanFilter:
    def __init__(
        self,
        x_0: np.ndarray,
        p: np.ndarray,
        phi: np.ndarray,
        gamma: np.ndarray,
        delta: np.ndarray,
        cv: np.ndarray,
        cw: np.ndarray
    ):
        """
        Kalman filter
        :param x_0: Initial state vector
        :param p: Initial covariance matrix on x_0
        :param phi: State transition matrix
        :param gamma: Command model matrix
        :param delta: Sensors model matrix
        :param cv: Variance on step
        :param cw: Variance on sensors
        """
        self._x_hat = x_0
        self._num_states, _ = self._x_hat.shape
        self._p = p
        self._phi = phi
        self._gamma = gamma
        self._delta = delta
        self._cv = cv
        self._cw = cw

    def step(self, u: np.ndarray):
        """
        Step the Kalman filter
        :param u: command vector
        :return: None
        """
        self._x_hat = self._phi @ self._x_hat + self._gamma @ u
        self._p = self._phi @ self._p @ self._phi.T + self._cv

    def update(self, measure: np.ndarray):
        """
        Update the Kalman filter with a measure
        :param measure: measurement made
        :return: None
        """
        z_hat = self._delta @ self._x_hat
        r = measure - z_hat
        k = self._p @ self._delta.T @ np.linalg.pinv(self._delta @ self._p @ self._delta.T + self._cw)
        self._x_hat = self._x_hat + k @ r
        self._p = (np.identity(self._num_states) - k @ self._delta) @ self._p

    @property
    def x_hat(self):
        return self._x_hat
