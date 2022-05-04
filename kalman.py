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

    def update(self, measure):
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

    dt = 1
    x_0 = np.array([[0], [0]])
    p = np.array([[1, 0], [0, 1]]) * 1
    phi = np.array([[1, dt], [0, 1]])
    gamma = np.array([[0.5 * dt ** 2], [dt]])
    delta = np.array([[1, 0]])
    cv = np.array([[2, 0], [0, 2]])
    cw = np.array([[30 ** 2]])
    kf = KalmanFilter(x_0, p, phi, gamma, delta, cv, cw)

    true_position = 0
    true_speed = 0
    nb_steps = 200
    time = 0

    true_positions = []
    true_speeds = []
    measurements = []
    estimates = []

    for i in range(nb_steps):
        time += dt
        command = np.array([[0.5 * np.cos(time / (nb_steps * dt / 2) * 2 * np.pi)]])

        # update true localization
        noised_command = command + np.random.randn() * 0.2 * np.abs(np.linalg.norm(command))
        true_position = true_position + true_speed * dt + 0.5 * noised_command * dt ** 2
        true_speed = true_speed + noised_command * dt

        measurement = true_position + np.random.randn() * 25
        kf.step(command)
        kf.update(measurement)

        true_positions.append(true_position)
        true_speeds.append(true_speed)
        measurements.append(measurement)
        estimates.append(kf.x_hat)

    true_positions = np.array(true_positions).squeeze()
    true_speeds = np.array(true_speeds).squeeze()
    measurements = np.array(measurements).squeeze()
    estimates = np.array(estimates).squeeze()


    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    def exponential_average(a, w):
        out = np.zeros_like(a)
        out[0] = a[0]
        for i in range(1, len(a)):
            out[i] = (1 - w) * out[i - 1] + w * a[i]
        return out


    m_average = moving_average(measurements, n=5)
    exp_average = exponential_average(measurements, 0.3)

    plt.plot(range(nb_steps), true_positions, label='True position')
    plt.scatter(range(nb_steps), measurements, marker='x', label='Measurements')
    plt.plot(range(nb_steps), estimates[:, 0], label='Kalman filter')
    plt.plot(range(len(m_average)), m_average, label='Moving average')
    plt.plot(range(len(exp_average)), exp_average, label='Exponential average')
    plt.title('Kalman filter')
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend()
    plt.show()
