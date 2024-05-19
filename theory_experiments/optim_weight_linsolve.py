from itertools import product
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from tqdm import tqdm

from verify_joan import HeadVsTarget, GegenbauerTransform, GegenbauerInverseTransform


def samples_from_sphere(dim, num_points, dtype=np.float64):
    X = np.random.randn(dim, num_points).astype(dtype)
    # normalize each row
    return X / np.linalg.norm(X, axis=0)


def clip1(x):
    return np.clip(x, -1, 1)


def error(a, b, C):
    return 1 - 2 * (b @ a) + a @ (C @ a)


def b_vec(Qs, Ks):
    dim, _ = Qs.shape
    approx = HeadVsTarget(dim, 50)
    angles = np.arccos(clip1((Qs * Ks).sum(axis=0)))
    return approx(angles).astype(np.float16)


def C_mat(Qs, Ks):
    """Assumes q = k
    """
    return (1 / 2) + (2 / (np.pi**2)) * (
        np.arcsin(clip1(Qs.T @ Qs)) * np.arcsin(clip1(Ks.T @ Ks))
    )


class COperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, Qs, Ks):
        assert Qs.shape == Ks.shape
        self.Qs = Qs
        self.Ks = Ks
        max_block = 2**27
        if self.H > (max_block):
            self.block_size = 1
        else:
            self.block_size = min((max_block) // self.H, self.H)
        self.num_blocks = self.H // self.block_size

    @property
    def dim(self):
        return self.Qs.shape[0]

    @property
    def H(self):
        return self.Qs.shape[1]

    @property
    def shape(self):
        return (self.H, self.H)

    @property
    def dtype(self):
        return np.result_type(self.Qs, self.Ks)

    @staticmethod
    def synthesize_C(QTQ, KTK):
        return (1 / 2) + (2 / (np.pi**2)) * (
            np.arcsin(clip1(QTQ))
            * np.arcsin(clip1(KTK))
        )

    def matvec(self, x):
        result = np.empty(self.H)
        for block_ix in range(self.num_blocks):
            Ci = self.synthesize_C(
                self.Qs.T[
                    (block_ix * self.block_size) : ((block_ix + 1) * self.block_size), :
                ]
                @ self.Qs,
                self.Ks.T[
                    (block_ix * self.block_size) : ((block_ix + 1) * self.block_size), :
                ]
                @ self.Ks
            )
            result[(block_ix * self.block_size) : ((block_ix + 1) * self.block_size)] = Ci @ x
        return result


class Callback:
    def __init__(self):
        self.num_iter = 0
        print()

    def __call__(self, _):
        self.num_iter += 1
        print(self.num_iter, end=" ", flush=True)


def construction_error(Qs, Ks):
    b = b_vec(Qs, Ks)

    # C = COperator(Qs, Ks)
    # a, info = scipy.sparse.linalg.cg(C, b)  # callback=Callback()

    C = C_mat(Qs, Ks)
    # print(C.nbytes)
    a, _ = scipy.sparse.linalg.cg(C, b)
    # a = scipy.linalg.solve(C, b, assume_a='pos')

    return error(a, b, C)


def num_harmonic(d, l):
    return ((2*l + d - 2)/l) * comb(l+d-3, l-1)


# TODO: instead of saving the Gegenbauer expansion of the pdf,
# integrate each term to get the cdf, which is easy to sample
class SmartDistribution(scipy.stats.rv_continuous):
    def __init__(self, dim, num_terms=50):
        super().__init__(a=-np.pi, b=np.pi)
        signG = GegenbauerTransform(dim, np.sign, 'odd')
        arcsinG = GegenbauerTransform(dim, np.arcsin, 'odd')
        self._weight_fun = GegenbauerInverseTransform(
            dim,
            np.array([
                (
                    (np.pi / 2)
                    * (signG.coeff(deg) / arcsinG.coeff(deg))
                    * num_harmonic(dim, deg)
                    if (deg % 2 == 1)
                    else 0
                )
                for deg in range(num_terms)
            ]),
        )
        theta_grid = np.linspace(-np.pi, np.pi, 10_000)
        self.normalization = sum(np.abs(self._weight_fun(np.cos(theta_grid)))) * (2 * np.pi / len(theta_grid))

        # xxx = np.linspace(-1, 1, 10000)
        # plt.plot(xxx, np.abs(u(xxx)))
        # print(sum(np.abs(xxx))/1000)

    def _pdf(self, theta):
        return np.abs(self._weight_fun(np.cos(theta))) / self.normalization


def euql_construction(dim, H):
    # Note: on its own, this fails completely
    Qs = samples_from_sphere(dim, H)
    return Qs, Qs


def gaussian_construction(dim, H):
    Qs = samples_from_sphere(dim, H)
    Ks = samples_from_sphere(dim, H)
    return Qs, Ks


def random_qk(dim, H, theta):
    Qs = samples_from_sphere(dim, H)
    e2s = samples_from_sphere(dim, H)
    e2s -= (Qs * e2s).sum(axis=0) * Qs
    Ks = np.cos(theta) * Qs + np.sin(theta) * e2s
    return Qs, Ks


def uniform_construction(dim, H):
    return random_qk(dim, H, np.arccos(np.random.uniform(-1, 1, H)))


if __name__ == "__main__":
    dims = np.array([2**i for i in range(3, 7)])
    Hs = np.array([2**i for i in range(15)])
    ntrials = 5

    results = []
    for (_, dim, H) in tqdm(list(product(range(ntrials), dims, Hs))):
        results.append(
            dict(
                dim=dim,
                H=H,
                rank=1,
                mse=construction_error(*uniform_construction(dim, H)),
                distribution="compromise",
            )
        )

    df = pd.DataFrame(results).rename({
        "dim": "Dimension",
        "H": "Number of Heads",
        "mse": "Mean Squared Error"
    }, axis=1)
    df = df[df["distribution"] == "compromise"]
    df["1/MSE"] = 1/df["Mean Squared Error"]
    sns.lineplot(
        x="Number of Heads",
        y="Mean Squared Error",
        data=df,
        hue="Dimension",
        color="Dimension",
        errorbar="pi",
    )
    plt.xscale("log")
    plt.savefig("paper_experiments/imgs/random_features.png", dpi=500)


# if __name__ == "__main__":
#     dim = 32
#     distribution = SmartDistribution(dim)
#     xxx = np.linspace(-np.pi, np.pi, 10_000)
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['font.size'] = 14
#     plt.figure(figsize=(7.5, 2.5), dpi=500)
#     plt.plot(xxx, distribution.pdf(xxx))
#     plt.xlabel("$\cos(\mathbf{q}^\\top \mathbf{k})$")
#     plt.ylabel("$\left|u(\mathbf{q}^\\top \mathbf{k})\\right|$")
