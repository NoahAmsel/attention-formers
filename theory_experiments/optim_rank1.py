from abc import ABC, abstractmethod
from math import sqrt

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def samples_from_sphere(dim, num_points, dtype=None):
    X = torch.randn((dim, num_points), dtype=dtype)
    # normalize each row
    return X / X.norm(dim=0)


def angle_between(x):
    """Takes an angle and normalizes it to the range [0, pi]
    """
    return torch.pi - torch.abs(torch.remainder(x, 2*torch.pi) - torch.pi)


def plot_angles(thetas):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_rticks([])
    for theta in thetas:
        ax.plot([theta, theta], [0, 1], c='blue')
        ax.plot([theta + torch.pi, theta + torch.pi], [0, 1], c='blue')
    return fig, ax


def plot_heads_3d(qs):
    ax = plt.figure().add_subplot(projection='3d')
    for h in range(qs.shape[1]):
        q = qs[:, h]
        qq = torch.stack([q, -q]).numpy()
        ax.plot(qq[:, 0], qq[:, 1], qq[:, 2])


class AbstractRank1(torch.nn.Module, ABC):
    @abstractmethod
    def dim(self):
        raise NotImplementedError

    def H(self):
        raise NotImplementedError

    def equalize_qk(self):
        # Note: this sets them equal initially but does not tie the weights together
        with torch.no_grad():
            self.qs.data = self.ks.data
        return self

    def mse(self):
        scale = 1/(4 * sqrt(self.dim()))
        b = scale * self.unscaled_edge_vector() + 0.5
        C = self.kernel() + 0.5
        return 1 - torch.inner(b, torch.linalg.solve(C, b))
        # return 1 - torch.inner(b, torch.linalg.lstsq(C, b).solution)

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
        self.qs = torch.nn.Parameter(samples_from_sphere(dim, H, dtype=torch.double))
        self.ks = torch.nn.Parameter(samples_from_sphere(dim, H, dtype=torch.double))
        self.clipping = clipping

    def renormalize(self):
        self.qs /= torch.norm(self.qs, dim=0)
        self.ks /= torch.norm(self.ks, dim=0)

    def dim(self):
        return self.qs.shape[0]

    def kernel(self):
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
        assert False, "unknown"
        # this is wrong
        # return (self.qs * self.ks).sum(dim=0)


class Rank1Angle(AbstractRank1):
    def __init__(self, H):
        super().__init__()
        # entries of qs and ks always lie in [0, 2pi)
        self.qs = torch.nn.Parameter((2 * torch.pi) * torch.rand(H, dtype=torch.double))
        self.ks = torch.nn.Parameter((2 * torch.pi) * torch.rand(H, dtype=torch.double))

    @classmethod
    def equispace(cls, H):
        model = cls(H)
        model.qs = torch.nn.Parameter(torch.linspace(0, torch.pi*(1-1/H), H, dtype=torch.double))
        model.ks = torch.nn.Parameter(torch.linspace(0, torch.pi*(1-1/H), H, dtype=torch.double))
        return model

    def renormalize(self):
        # Do we even need to renormalize?
        # self.actually_renormalize()
        pass

    def actually_renormalize(self):
        self.qs.data = torch.remainder(self.qs, 2*torch.pi)
        self.ks.data = torch.remainder(self.ks, 2*torch.pi)

    def snap(self, factor=24):
        with torch.no_grad():
            self.qs -= self.qs[0].clone()
            self.ks -= self.ks[0].clone()
            self.actually_renormalize()
            self.qs.data = (self.qs.data / (torch.pi / factor)).round() *  (torch.pi / factor)
            self.ks.data = (self.ks.data / (torch.pi / factor)).round() *  (torch.pi / factor)

    def dim(self):
        return 2
    
    def H(self):
        return len(self.qs)

    @staticmethod
    def arcsin_cos(x):
        return torch.pi/2 - angle_between(x)

    def kernel(self):
        return (
            (2 / torch.pi**2)
            * self.arcsin_cos(self.qs.reshape((-1, 1)) - self.qs.reshape((1, -1)))
            * self.arcsin_cos(self.ks.reshape((-1, 1)) - self.ks.reshape((1, -1)))
        )

    def edge_vector(self):
        theta = angle_between(self.qs - self.ks)
        fixed_theta = torch.pi/2 - torch.abs(torch.pi/2 - theta)
        return torch.sign(torch.pi/2 - theta) * (0.25 - (fixed_theta/torch.pi)**2)

    def mse(self):
        b = 0.5 + self.edge_vector()
        C = 0.5 + self.kernel()
        return 1 - torch.inner(b, torch.linalg.solve(C, b))
        # return 1 - torch.inner(b, torch.linalg.lstsq(C, b).solution)


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


if __name__ == "__main__":
    # # Quick and dirty experiments
    # print([Rank1(dim=2, H=2**i, clipping=1).mse().data for i in range(11)])
    # print([Rank1(dim=2, H=2**i, clipping=1).equalize_qk().mse().data for i in range(11)])


    # model = Rank1(dim=2, H=2**0, clipping=.9999)
    model = Rank1Angle(H=40)
    # model = model.cuda()
    model.equalize_qk()
    print(f"Before: {model.mse():.5f}")
    train(model, steps=10_000, lr=1)
    train(model, steps=10_000, lr=.1)
    train(model, steps=10_000, lr=.01)
    train(model, steps=10_000, lr=.001)
    model.snap(2 * model.H())  # is this right? maybe just H is enough?
    print(f"After: {model.mse():.5f}")
    print(f"qs\n{(model.qs * 180 / torch.pi).data}")
    print(f"ks\n{(model.ks * 180 / torch.pi).data}")
    print(f"qk angles:\n{((180/torch.pi) * torch.arccos(model.unscaled_edge_vector()).data).round()}")
    # print(f"qq angles:\n{((180/torch.pi) * angle_between(model.qs.reshape((-1, 1)) - model.qs.reshape((1, -1)))).round().data}")
    # print(f"kk angles:\n{((180/torch.pi) * angle_between(model.ks.reshape((-1, 1)) - model.ks.reshape((1, -1)))).round().data}")

    # TODO
    # i was trying rerunning stuff in double precision. also remember to switch between least squares and linear solver
    # also try playing with equalize_qk. how do we get it to initialize them to be the same, and how do we get it to guarantee that they're the same
    # so far for H=4 it seems that equalizing gets you to a better minimum then you tend to get otherwise. there are local minima where q,k are antiparallel but they aren't as good

    plot_angles(model.qs.data.detach())
