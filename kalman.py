import numpy as np


class KalmanFilter:
    def __init__(self, x_0: np.ndarray, p: np.ndarray, phi: np.ndarray, gamma: np.ndarray, delta: np.ndarray,
                 cv: np.ndarray,
                 cw: np.ndarray):
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

    def measure(self, measure):
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x_0 = np.array([[0]])
    p = np.array([[1]]) * 1
    phi = np.array([[1]])
    gamma = np.array([[1]])
    delta = np.array([[1]])
    cv = np.array([[1]])
    cw = np.array([[30 ** 2]])
    kf = KalmanFilter(x_0, p, phi, gamma, delta, cv, cw)

    true_position = 0
    nb_steps = 100
    step_size = 1

    true_positions = []
    measurements = []
    estimates = []

    for i in range(nb_steps):
        true_position += step_size + np.random.randn()
        measurement = true_position + np.random.randn() * 25
        kf.step(np.array([[step_size]]))
        kf.measure(measurement)

        true_positions.append(true_position)
        measurements.append(measurement)
        estimates.append(kf.x_hat)

    estimates = np.array(estimates).squeeze()


    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    m_average = moving_average(measurements, n=5)

    plt.plot(range(nb_steps), true_positions, label='true_positions')
    plt.scatter(range(nb_steps), measurements, marker='x', label='measurements')
    plt.plot(range(nb_steps), estimates, label='estimates')
    plt.plot(range(len(m_average)), m_average, label='moving average')
    plt.title('Kalman filter')
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend()
    plt.show()
