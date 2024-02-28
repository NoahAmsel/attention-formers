import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def samples_from_sphere(dim, num_points):
    X = np.random.randn(dim, num_points)
    # normalize each row
    return X / np.linalg.norm(X, axis=0)


def slope(X, Y):
    return ((X*Y).mean() - X.mean()*Y.mean()) / ((X**2).mean() - (X.mean())**2)


def angle_between(x):
    """Takes an angle and normalizes it to the range [0, pi]
    """
    return np.pi - np.abs(np.remainder(x, 2*np.pi) - np.pi)


def qk(dim, theta):
    q = np.zeros(dim)
    q[0] = 1
    k = np.zeros(dim)
    k[:2] = [np.cos(theta), np.sin(theta)]
    return q, k


def random_qk(dim, theta):
    # TODO: vectorize this so it can quickly generate many samples of (q,k)
    if theta == 0:
        # for speed, this is a seperate case
        q = samples_from_sphere(dim, 1).squeeze()
        return q, q
    else:
        r = samples_from_sphere(dim, 2)
        q = r[:, 0]
        e2 = r[:, 1] - np.inner(q, r[:, 1]) * q
        k = np.cos(theta) * q + np.sin(theta) * e2
        return q, k


def estimate_head_vs_target(dim, theta, ntrials=100_000):
    """For a head with angle(q,k) = theta,
    estimate the probability it equals the target
    over uniform y and uniform and perpendicular x1, x2
    which is Pr_{x,y}[<x,y><x,k><q,y> > 0]
    """
    q, k = qk(dim, theta)
    count = 0
    for _ in tqdm(range(ntrials)):
        r = np.random.randn(dim, 2)  # norm doesn't matter
        x, y = r[:, 0], r[:, 1]
        if np.inner(x, y) * np.inner(x, k) * np.inner(q, y) > 0:
            count += 1
    return count / ntrials


def estimate_head_vs_head(q, k, q2, k2, ntrials=100_000, tqdm=False):
    """For heads (q,k) and (q2, k2),
    estimate the probability they equal each other
    over uniform y and uniform and perpendicular x1, x2
    which is Pr_{x,y}[<x,k><q,y><x,k'><q',y> > 0]
    """
    assert q.shape == k.shape == q2.shape == k2.shape
    dim = len(q)
    count = 0
    rg = range(ntrials)
    if tqdm:
        rg = tqdm(rg)
    for _ in rg:
        r = np.random.randn(dim, 2)  # norm doesn't matter
        x, y = r[:, 0], r[:, 1]
        if np.inner(x, k) * np.inner(q, y) * np.inner(x, k2) * np.inner(q2, y) > 0:
            count += 1
    return count / ntrials


def alt_estimate_head_vs_random_head(dim, theta1, theta2, ntrials=100_000):
    """For heads (q,k) and (q2, k2) drawn uniformly at random
    conditioned on angle(q,k) = theta1, angle(q2,k2) = theta2
    estimates the probability that thtey equal each other
    """
    q, k = qk(dim, theta1)
    count = 0
    for _ in tqdm(range(ntrials)):
        q2, k2 = random_qk(dim, theta2)
        count += estimate_head_vs_head(q, k, q2, k2, ntrials=1)
    return count / ntrials


def clip_asin(x):
    return np.arcsin(np.clip(x, -1, 1))


def head_vs_head(q, k, q2, k2):
    qq2 = np.inner(q, q2) / (np.linalg.norm(q) * np.linalg.norm(q2))
    kk2 = np.inner(k, k2) / (np.linalg.norm(k) * np.linalg.norm(k2))
    return (1/2) + (2 / np.pi**2) * clip_asin(qq2) * clip_asin(kk2)


def estimate_head_vs_random_head(dim, theta1, theta2, ntrials=100_000):
    q, k = qk(dim, theta1)
    return sum(head_vs_head(q, k, *random_qk(dim, theta2)) for _ in tqdm(range(ntrials))) / ntrials


##############
# 2D
##############


def estimate_head_vs_random_head_2D(theta1, theta2, ntrials=100_000):
    """Approximates E[arcsin(<q,q'>) * arcsin(<k,k'>)]
    where angle between q and k is theta1
    q' is drawn uniformly, and angle between q' and k' is theta2
    """
    def result(k_prime):
        return np.inner(
            np.pi/2 - angle_between(q_prime),
            np.pi/2 - angle_between(k_prime - theta1) / ntrials
        )
    q_prime = np.linspace(0, 2*np.pi, ntrials, endpoint=False)
    return (1/2) + (2 / np.pi**2) * (result(q_prime + theta2) + result(q_prime - theta2)) / 2


def head_vs_head_2D(q, k, q2, k2):
    q_angle = angle_between(q - q2)
    k_angle = angle_between(k - k2)
    return (1/2) + (2 / np.pi**2) * (np.pi/2 - q_angle) * (np.pi/2 - k_angle)


def head_vs_random_head_2D(theta1, theta2):
    a = angle_between(theta1 - theta2) / np.pi
    b = angle_between(theta1 + theta2) / np.pi
    return (4 + 2*(a**3 + b**3) - 3*(a**2 + b**2)) / 6


def head_vs_target_2D(theta):
    # this simplifies significantly when theta < pi/2
    # it's a differentiable, piecewise quadratic with a knot at pi/2
    fixed_theta = np.pi/2 - np.abs(np.pi/2 - theta)
    return (1/2) + np.sign(np.pi/2 - theta) * (0.25 - (fixed_theta/np.pi)**2)


def MSE_weighted_random_2D(H):
    q_angles = 2*np.pi * np.random.rand(H)
    k_angles = 2*np.pi * np.random.rand(H)
    C = (1 / 2) + (2 / np.pi**2) * (
        np.pi / 2 - angle_between(q_angles.reshape((-1, 1)) - q_angles.reshape((1, -1)))
    ) * (
        np.pi / 2 - angle_between(k_angles.reshape((-1, 1)) - k_angles.reshape((1, -1)))
    )
    b = head_vs_target_2D(angle_between(q_angles - k_angles))
    return 1 - np.inner(b, np.linalg.solve(C, b))


##############
# Experiments
##############


if __name__ == "__main__":
    # Experiment 1: head vs target in 2D
    thetas = np.linspace(0, np.pi, 50, endpoint=True)
    estimate = [estimate_head_vs_target(2, theta) for theta in thetas]
    exact = head_vs_target_2D(thetas)
    plt.plot(thetas, exact, thetas, estimate)
    plt.plot(exact, estimate)


if __name__ == "__main__":
    # Experiment 2: head vs head 4D
    q, k = qk(4, np.pi/4)
    theta2s = np.linspace(0, np.pi, 20, endpoint=True)
    qk2s = [random_qk(4, theta2) for theta2 in theta2s for _ in range(3)]
    estimates = np.array([estimate_head_vs_head(q, k, q2, k2, ntrials=100_000) for (q2, k2) in tqdm(qk2s)])
    exact = np.array([head_vs_head(q, k, q2, k2) for (q2, k2) in qk2s])
    plt.plot((estimates - exact)/exact)


if __name__ == "__main__":
    # Experiment 3: head vs head 2D
    thetas = np.linspace(0, np.pi, 50, endpoint=True)
    estimates = np.array([[estimate_head_vs_random_head_2D(theta1, theta2) for theta2 in thetas] for theta1 in thetas])
    exact = np.array([[head_vs_random_head_2D(theta1, theta2) for theta2 in thetas] for theta1 in thetas])
    plt.plot(thetas, estimates[:, [0, 15, 30]], thetas, exact[:, [0, 15, 30]])
    np.max(np.abs(estimates - exact)/exact)


if __name__ == "__main__":
    Hs = np.array([2**i for i in range(15)])
    errors = np.array([MSE_weighted_random_2D(H) for H in Hs])
    plt.loglog(Hs, errors)
    plt.xlabel("H")
    plt.ylabel("1 - b^TC^{-1}b")

    slope(np.log(Hs), np.log(errors))
