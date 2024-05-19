import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def samples_from_sphere(dim, num_points):
    X = np.random.randn(dim, num_points)
    # normalize each row
    return X / np.linalg.norm(X, axis=0)


def random_qk(dim, theta, num):
    qs = samples_from_sphere(dim, num)
    e2 = samples_from_sphere(dim, num)
    e2 -= (qs * e2).sum(axis=0) * qs
    ks = np.cos(theta) * qs + np.sin(theta) * e2
    return qs, ks


def p_head_correct(dim, theta_wy, theta_qk=None, nsamples=100_000):
    # implicitly, w = e1, y = cos(theta) e1 + sin(theta) e2
    true_sign = np.sign(np.cos(theta_wy))
    if theta_qk is None:
        # totally independent
        qs = samples_from_sphere(dim, nsamples)
        ks = samples_from_sphere(dim, nsamples)
    else:
        qs, ks = random_qk(dim, theta_qk, nsamples)
    w_qs = qs[0, :]
    y_ks = ks[0, :] * np.cos(theta_wy) + ks[1, :] * np.sin(theta_wy)
    return np.mean(np.sign(w_qs * y_ks) == true_sign)


dim = 32
theta_qk = 0
thetas = np.linspace(0, np.pi, 100)
emp = np.array([[p_head_correct(dim, theta, theta_qk=theta_qk, nsamples=100_000) for theta in tqdm(thetas)] for theta_qk in [0, np.pi/4, np.pi/3, np.pi/2]])
gilad = 0.5 + np.abs(np.cos(thetas))
# noah = np.abs(thetas / np.pi - 0.5) + 0.5
noah = 1 - np.arccos(np.abs(np.cos(thetas)))/np.pi
plt.plot(thetas, emp.T)
plt.plot(thetas, noah)
plt.plot(thetas, gilad)
plt.title("Probability of a head q,k being correct")
plt.xlabel("angle between x1 - x2 and y")
plt.legend(["0", "pi/4", "pi/3", "pi/2"], title="angle btwn q,k")
