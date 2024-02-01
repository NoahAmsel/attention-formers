from math import sqrt

import torch
from tqdm import tqdm


def samples_from_sphere(dim, num_points):
    X = torch.randn((dim, num_points))
    # normalize each row
    return X / X.norm(dim=0)


class Rank1(torch.nn.Module):
    def __init__(self, dim, H, clipping=1.0):
        super().__init__()
        # columns of qs and ks are always rank 1
        self.qs = torch.nn.Parameter(samples_from_sphere(dim, H))
        self.ks = torch.nn.Parameter(samples_from_sphere(dim, H))
        self.clipping = clipping

    def equalize_qk(self):
        with torch.no_grad():
            self.ks.data = self.qs.data
        return self

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

    def mse(self):
        scale = 1/(4 * sqrt(self.dim()))
        b = scale * self.unscaled_edge_vector() + 0.5
        C = self.clipped_kernel() + 0.5
        return 1 - torch.inner(b, torch.linalg.solve(C, b))

    def dKd(self):
        scale = 1 / (4 * sqrt(self.dim()))
        d = scale * self.unscaled_edge_vector()
        K = self.clipped_kernel()
        return torch.inner(d, torch.linalg.solve(K, d))
        # return -torch.inner(d, torch.linalg.lstsq(K, d).solution)


def train(model, steps):
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    for _ in tqdm(range(steps)):

        def closure():
            optimizer.zero_grad()
            loss = model.mse()
            loss.backward(retain_graph=False)
            return loss

        # GD step
        optimizer.step(closure)

        # project
        with torch.no_grad():
            model.renormalize()


# # Quick and dirty experiments
# print([Rank1(dim=2, H=2**i, clipping=1).mse().data for i in range(11)])
# print([Rank1(dim=2, H=2**i, clipping=1).equalize_qk().mse().data for i in range(11)])


model = Rank1(dim=2, H=2**4, clipping=.9999)
# model.equalize_qk()
print(f"Before: {model.mse():.5f}")
train(model, 1000)
print(f"After: {model.mse():.5f}")
print(f"qk angles:\n{model.unscaled_edge_vector()}")
