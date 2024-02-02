from abc import ABC, abstractmethod
from math import sqrt

import torch
from tqdm import tqdm


def samples_from_sphere(dim, num_points):
    X = torch.randn((dim, num_points))
    # normalize each row
    return X / X.norm(dim=0)


def angle_between(x):
    """Takes an angle and normalizes it to the range [0, pi]
    """
    return torch.pi - torch.abs(torch.remainder(x, 2*torch.pi) - torch.pi)


class AbstractRank1(torch.nn.Module, ABC):
    @abstractmethod
    def dim(self):
        raise NotImplementedError

    def equalize_qk(self):
        with torch.no_grad():
            self.qs.data = self.ks.data
        return self

    def mse(self):
        scale = 1/(4 * sqrt(self.dim()))
        b = scale * self.unscaled_edge_vector() + 0.5
        C = self.kernel() + 0.5
        return 1 - torch.inner(b, torch.linalg.solve(C, b))

    def dKd(self):
        scale = 1 / (4 * sqrt(self.dim()))
        d = scale * self.unscaled_edge_vector()
        K = self.kernel()
        return torch.inner(d, torch.linalg.solve(K, d))
        # return -torch.inner(d, torch.linalg.lstsq(K, d).solution)


class Rank1(AbstractRank1):
    def __init__(self, dim, H, clipping=1.0):
        super().__init__()
        # columns of qs and ks are always rank 1
        self.qs = torch.nn.Parameter(samples_from_sphere(dim, H))
        self.ks = torch.nn.Parameter(samples_from_sphere(dim, H))
        self.clipping = clipping

    def renormalize(self):
        self.qs /= torch.norm(self.qs, dim=0)
        self.ks /= torch.norm(self.ks, dim=0)

    def dim(self):
        return self.qs.shape[0]

    def clipped_kernel(self):
        return (
            (2 / torch.pi**2)
            * torch.arcsin(
                torch.clip(self.qs.T @ self.qs, -self.clipping, self.clipping)
            )
            * torch.arcsin(
                torch.clip(self.ks.T @ self.ks, -self.clipping, self.clipping)
            )
        )

    def unscaled_edge_vector(self):
        return (self.qs * self.ks).sum(dim=0)


class Rank1Angle(AbstractRank1):
    def __init__(self, H):
        super().__init__()
        # entries of qs and ks always lie in [0, 2pi)
        self.qs = torch.nn.Parameter((2 * torch.pi) * torch.rand(H))
        self.ks = torch.nn.Parameter((2 * torch.pi) * torch.rand(H))

    # Do we even need to renormalize?
    def renormalize(self):
        self.qs.data = torch.remainder(self.qs, 2*torch.pi)
        self.ks.data = torch.remainder(self.ks, 2*torch.pi)

    def dim(self):
        return 2

    @staticmethod
    def arcsin_cos(x):
        return torch.pi/2 - angle_between(x)

    def kernel(self):
        return (
            (2 / torch.pi**2)
            * self.arcsin_cos(self.qs.reshape((-1, 1)) - self.qs.reshape((1, -1)))
            * self.arcsin_cos(self.ks.reshape((-1, 1)) - self.ks.reshape((1, -1)))
        )

    def unscaled_edge_vector(self):
        return torch.cos(self.qs - self.ks)


def train(model, steps, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    for _ in tqdm(range(steps)):
        def closure():
            optimizer.zero_grad()
            loss = model.mse()
            loss.backward(retain_graph=False)
            return loss

        # GD step
        optimizer.step(closure)
        # scheduler.step()

        # project
        with torch.no_grad():
            model.renormalize()


# # Quick and dirty experiments
# print([Rank1(dim=2, H=2**i, clipping=1).mse().data for i in range(11)])
# print([Rank1(dim=2, H=2**i, clipping=1).equalize_qk().mse().data for i in range(11)])


# model = Rank1(dim=2, H=2**0, clipping=.9999)
model = Rank1Angle(H=2)
# model = model.cuda()
# model.equalize_qk()
print(f"Before: {model.mse():.5f}")
train(model, steps=1_000, lr=1)
print(f"After: {model.mse():.5f}")
print(f"qk angles:\n{((180/torch.pi) * torch.arccos(model.unscaled_edge_vector()).data).round()}")
print(f"qq angles:\n{((180/torch.pi) * angle_between(model.qs.reshape((-1, 1)) - model.qs.reshape((1, -1)))).round().data}")
print(f"kk angles:\n{((180/torch.pi) * angle_between(model.ks.reshape((-1, 1)) - model.ks.reshape((1, -1)))).round().data}")
