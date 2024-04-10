import matplotlib.pyplot as plt
import torch
from torch_cg import CG  # https://github.com/sbarratt/torch_cg
from tqdm import tqdm

from verify_joan import HeadVsTarget


def samples_from_sphere(dim, num_points, dtype=None, device=None):
    X = torch.randn((dim, num_points), dtype=dtype, device=device)
    # normalize each row
    return X / X.norm(dim=0)


def clip1(x):
    return torch.clip(x, -1, 1)


def error(a, b, C):
    return 1 - 2 * (b @ a) + a @ (C.vector_mult(a))


def approx_head_vs_target(Qs, Ks, num_gegenbauer_terms):
    # Qs and Ks have shape dim x num heads
    q_norms_inv = 1 / torch.norm(Qs, dim=0)
    k_norms_inv = 1 / torch.norm(Ks, dim=0)
    angles = torch.arccos(torch.clip((Qs * Ks).sum(axis=0) * q_norms_inv * k_norms_inv, -1, 1))
    angles = angles.numpy(force=True)
    a = HeadVsTarget(Qs.shape[0], num_gegenbauer_terms)(angles)
    return torch.from_numpy(a).to(Qs.dtype).to(Qs.device)


def C_mat(Qs, Ks):
    """Assumes q = k
    """
    return (1 / 2) + (2 / (torch.pi**2)) * (
        torch.arcsin(clip1(Qs.T @ Qs)) * torch.arcsin(clip1(Ks.T @ Ks))
    )


class COperator:
    def __init__(self, Qs, Ks):
        assert Qs.shape == Ks.shape
        self.Qs = Qs
        self.Ks = Ks
        # for 128 GB, block = 2^30 suffices
        max_block = 2**28
        if self.H ** 2 <= max_block:
            self.cache_C = self.synthesize_C(
                self.Qs.T @ self.Qs,
                self.Ks.T @ self.Ks
            )
        else:
            print("C cannot fit in max_block")
            self.cache_C = None
            if self.H > max_block:
                self.block_size = 1
            else:
                self.block_size = min(max_block // self.H, self.H)
            self.num_blocks = self.H // self.block_size
            assert self.num_blocks * self.block_size == self.H

    @property
    def dim(self):
        return self.Qs.shape[0]

    @property
    def H(self):
        return self.Qs.shape[1]

    @property
    def shape(self):
        return (self.H, self.H)

    @staticmethod
    def synthesize_C(QTQ, KTK):
        return (1 / 2) + (2 / (torch.pi**2)) * (
            torch.arcsin(clip1(QTQ))
            * torch.arcsin(clip1(KTK))
        )

    def vector_mult(self, x):
        if self.cache_C is not None:
            return self.cache_C @ x

        result = torch.empty_like(x)
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

    def __call__(self, x):
        return self.vector_mult(x[0, :, 0]).reshape(1, -1, 1)


def construction_error(dim, H, rtol=1e-3, dtype=None, device=None):
    Qs = samples_from_sphere(dim, H, dtype=dtype, device=device)
    Ks = samples_from_sphere(dim, H, dtype=dtype, device=device)
    # Ks = Qs  # this doesn't work at all
    b = approx_head_vs_target(Qs, Ks, 50)
    # C = C_mat(Qs, Ks)
    C = COperator(Qs, Ks)
    # a = scipy.linalg.solve(C, b, assume_a='pos')
    a = CG(C, rtol=rtol, maxiter=100, verbose=True).forward(b.reshape(1, -1, 1))[0, :, 0]
    e = float(error(a, b, C))
    print(e)
    return e


if __name__ == "__main__":
    # assert torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 4  # MUST BE POWER OF 2
    Hs = torch.tensor([2**i for i in range(19)])
    errors = torch.tensor([construction_error(dim, H, rtol=1e-3, device=device, dtype=torch.float64) for H in tqdm(Hs)])
    print(errors)
    plt.plot(Hs, errors)
    plt.xscale("log")
    plt.title(f"Random features: dim={dim}")
    plt.xlabel("Number of Heads")
    plt.ylabel("MSE")

    plt.plot(Hs, 1/errors)
    plt.xscale("log")
    plt.title(f"Random features: dim={dim}")
    plt.xlabel("Number of Heads")
    plt.ylabel("1 / MSE")
