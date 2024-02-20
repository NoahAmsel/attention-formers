import flamp
import matplotlib.pyplot as plt
import numpy as np


def rescaled_operator(t1, t2):
    a = np.abs(t1 - t2)
    b = t1 + t2
    return 4 + 2 * a**3 + 2 * b**3 - 3 * a**2 - 3 * b**2


def operator(theta1, theta2):
    a = np.abs(theta1 - theta2) / pi
    b = (theta1 + theta2) / pi
    return (4 + 2*(a**3 + b**3) - 3*(a**2 + b**2)) / 6


def angle_between(x):
    """Takes an angle and normalizes it to the range [0, pi]
    """
    return pi - np.abs(np.remainder(x, 2*pi) - pi)


def wide_range_operator(theta1, theta2):
    a = angle_between(theta1 - theta2) / pi
    b = angle_between(theta1 + theta2) / pi
    return (4 + 2*(a**3 + b**3) - 3*(a**2 + b**2)) / 6


if __name__ == "__main__":
    # Experiment 1: solving for a()
    flamp.set_dps(100)
    N = 1001
    thetas = flamp.linspace(0, 1/2, N)
    A = rescaled_operator(thetas.reshape(-1, 1), thetas.reshape(1, -1))
    b = .75 - thetas ** 2
    x = flamp.cholesky_solve(A, b)

    print(np.max(np.abs(A @ x - b)))
    plt.figure()
    plt.plot(thetas, x)
    plt.show()


if __name__ == "__main__":
    # Experiment 2: solving for a() with proper scaling
    flamp.set_dps(100)
    pi = flamp.gmpy2.const_pi()
    N = 1001
    thetas = flamp.linspace(0, pi/2, N)
    A = operator(thetas.reshape(-1, 1), thetas.reshape(1, -1))
    b = .75 - (thetas/pi) ** 2
    x = flamp.cholesky_solve(A, b)
    print(1 - x @ b)

    print(np.max(np.abs(A @ x - b)))
    plt.figure()
    plt.plot(thetas, x)
    plt.show()


if __name__ == "__main__":
    # Experiment 3: solving for a() over full range
    flamp.set_dps(100)
    pi = flamp.gmpy2.const_pi()
    N = 1001
    thetas = flamp.linspace(0, pi, N)
    A = wide_range_operator(thetas.reshape(-1, 1), thetas.reshape(1, -1))
    b = .75 - (thetas/pi) ** 2
    # TODO: when we try cholesky, it says "not psd". why?
    x = flamp.qr_solve(A, b)

    print(np.max(np.abs(A @ x - b)))
    plt.figure()
    plt.plot(thetas, x)
    plt.show()
