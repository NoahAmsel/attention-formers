import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from verify_joan import HeadVsTarget


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


def construction_error(dim, H):
    Qs = samples_from_sphere(dim, H)
    Ks = samples_from_sphere(dim, H)
    # Ks = Qs  # this doesn't work at all
    b = b_vec(Qs, Ks)
    # C = C_mat(Qs, Ks)
    C = COperator(Qs, Ks)
    # a = scipy.linalg.solve(C, b, assume_a='pos')
    cc = Callback()
    a, info = scipy.sparse.linalg.cg(C, b, callback=cc)
    if info != 0:
        print(dim, H, info)
    return error(a, b, C)


if __name__ == "__main__":
    dim = 8  # MUST BE POWER OF 2
    Hs = np.array([2**i for i in range(18)])
    errors = np.array([construction_error(dim, H) for H in tqdm(Hs)])
    print(errors)
    plt.plot(Hs, errors)
    plt.xscale("log")
    plt.yscale("log")
