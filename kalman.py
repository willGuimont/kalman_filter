import numpy as np


class KalmanFilter:
    def __init__(self, x_0: np.ndarray, p: np.ndarray, phi: np.ndarray, gamma: np.ndarray, delta: np.ndarray,
                 cv: np.ndarray,
                 cw: np.ndarray):
        self._x_hat = x_0
        self._num_states, _ = self._x_hat.shape
        self._p = p
        self._phi = phi
        self._gamma = gamma
        self._delta = delta
        self._cv = cv
        self._cw = cw

    def step(self, u, measure=None):
        self._x_hat = self._phi @ self._x_hat + self._gamma @ u
        self._p = self._phi @ self._p @ self._phi.T + self._cv
        if measure is not None:
            z_hat = self._delta @ self._x_hat
            r = measure - z_hat
            k = self._p @ self._delta.T @ np.linalg.pinv(self._delta @ self._p @ self._delta.T + self._cw)
            self._x_hat = self._x_hat + k @ r
            self._p = (np.identity(self._num_states) - k @ self._delta) @ self._p

    @property
    def x_hat(self):
        return self._x_hat


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    positions = list(range(100))
    kf = KalmanFilter(np.array([[0, 0]]).T,
                      np.ones((2, 2)) * 5,
                      np.array([[1, 1], [0, 0]]),
                      np.array([[0, 0]]).T,
                      np.array([[1, 0]]),
                      np.array([[1, 0]]).T,
                      np.array([[20]]))
    estimates = []
    measures = []
    for p in positions:
        p += np.random.randn() * 5
        measures.append(p)
        kf.step(np.array([1]), p)
        estimates.append(kf.x_hat)

    plt.plot(positions)
    plt.plot(list(map(lambda x: x[0], estimates)))
    plt.scatter(range(len(measures)), measures)
    plt.show()
