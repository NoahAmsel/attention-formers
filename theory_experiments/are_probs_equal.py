import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def unif(d, n):
    X = np.random.randn(d, n)
    X /= np.linalg.norm(X, axis=0)
    return X


def qk(dim, theta):
    q = np.zeros(dim)
    q[0] = 1
    k = np.zeros(dim)
    k[:2] = [np.cos(theta), np.sin(theta)]
    return q, k


def p1p2(dim, theta, ntrials=100_000):
    q, k = qk(dim, theta)
    count1 = 0
    count2 = 0
    for _ in tqdm(range(ntrials)):
        r = np.random.randn(dim, 3)
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        common = np.inner(x, k) * np.inner(q, y)
        if common * np.inner(x, z) * np.inner(z, y) > 0:
            count1 += 1
        if common * np.inner(x, y) > 0:
            count2 += 1
    return count1 / ntrials, count2 / ntrials


def clip_asin(x):
    return np.arcsin(np.clip(x, -1, 1))


def exp_kernel(dim, theta, ntrials=100_000):
    q, k = qk(dim, theta)
    Z = unif(dim, ntrials)
    return np.inner(clip_asin(Z.T @ q), clip_asin(Z.T @ k) / ntrials)


# # Experiment 1
# thetas = np.linspace(0, np.pi, 20)
# results = np.array([p1p2(20, theta, ntrials=int(1e6)) for theta in thetas])

# plt.plot(thetas, results)
# plt.plot(thetas, results - 0.5)


# Experiment 2
thetas = np.linspace(0, np.pi/2, 25, endpoint=False)
results = np.array([exp_kernel(2, theta, ntrials=int(1e7)) for theta in tqdm(thetas)])
plt.plot(thetas, results / np.cos(thetas))
plt.plot(thetas, results/results[0], thetas, np.cos(thetas))

results_big = np.array([exp_kernel(2, theta, ntrials=int(1e9)) for theta in tqdm(thetas)])
plt.plot(thetas, results_big / np.cos(thetas))
plt.plot(thetas, results_big/results_big[0], thetas, np.cos(thetas))
