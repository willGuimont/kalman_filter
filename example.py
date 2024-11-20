from kalman import KalmanFilter

import matplotlib.pyplot as plt
import numpy as np


dt = 1
x_0 = np.array([[0], [0]])
p = np.array([[1, 0], [0, 1]]) * 1
phi = np.array([[1, dt], [0, 1]])
gamma = np.array([[0.5 * dt ** 2], [dt]])
delta = np.array([[1, 0]])
cv = np.array([[0, 0], [0, 0.1]])
cw = np.array([[35 ** 2]])

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
plt.title('Position estimate')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.show()

plt.plot(range(nb_steps), true_speeds, label='True speed')
plt.plot(range(nb_steps), estimates[:, 1], label='Kalman filter')
plt.title('Speed estimate')
plt.xlabel('Time')
plt.ylabel('Speed')
plt.legend()
plt.show()
