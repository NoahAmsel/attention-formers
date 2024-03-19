from abc import ABC, abstractmethod
from math import sqrt

import torch
import scipy.linalg

from task import samples_from_sphere, create_rng, NearestPointDataset
from theory_experiments.verify_joan import HeadVsTarget

# TODO: add seed
class AbstractMultiheadAttention(torch.nn.Module, ABC):
    def __init__(self, dim, rank, nheads, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.nheads = nheads
        self.Q = torch.nn.Parameter(torch.empty((nheads, rank, dim), device=device, dtype=dtype))
        self.K = torch.nn.Parameter(torch.empty((nheads, rank, dim), device=device, dtype=dtype))
        self.VO = torch.nn.Parameter(torch.empty((nheads, dim, dim), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        """Pytorch initializes using Xavier initialization, where the fan-in / fan-out are calculated as follows
        - combined Q matrix of dimension dim x dim
        - combined K matrix of dimension dim x dim
        - combined V matrix of dimension dim x dim
        - O matrix of dimension dim x dim

        For Q, K, and V, it's really (nheads * rank) x dim, since they set rank = dim / nheads.
        But it seems wrong that the scale of Q and K should change based on the number of heads.
        Also we have a combined VO matrix of dimension dim x (nheads * dim) which *should* change with nheads
        """
        QK_a = sqrt(6.0 / (self.rank + self.dim))
        VO_a = sqrt(6.0 / (self.nheads * self.dim + self.dim))
        torch.nn.init.uniform_(self.Q, -QK_a, QK_a)
        torch.nn.init.uniform_(self.K, -QK_a, QK_a)
        torch.nn.init.uniform_(self.VO, -VO_a, VO_a)
        # torch.nn.init.xavier_uniform_(self.Q)
        # torch.nn.init.xavier_uniform_(self.K)
        # torch.nn.init.xavier_uniform_(self.VO)        

    def assemble_QK(self, head_ix):
        return self.Q.data[head_ix, :, :].T @ self.K.data[head_ix, :, :]

    def assemble_VO(self, head_ix):
        return self.VO.data[head_ix, :, :]

    def inside_heads(self, X, Y):
        # X is batch_size, dim, num keys
        # Y is batch_size, dim, num queries
        # self.Q and self.K are each nheads by rank by dim
        # output (attention matrices) is batch_size, num queries, num heads, num keys
        return torch.einsum("bdq,hrd,hre,bek->bqhk", Y, self.Q, self.K, X) / sqrt(self.dim)

    @abstractmethod
    def forward(self, X, Y):
        raise NotImplementedError

    @classmethod
    def perfect_construction(cls, dim, temperature=1, device=None, dtype=None):
        model = cls(dim=dim, rank=dim, nheads=1, device=device, dtype=dtype)
        model.Q.data[0, :, :] = temperature * torch.eye(model.dim, device=device, dtype=dtype)
        model.K.data[0, :, :] = torch.eye(model.dim, device=device, dtype=dtype)
        model.VO.data[0, :, :] = torch.eye(model.dim, device=device, dtype=dtype)
        return model

    @classmethod
    def random_construction(cls, dim, rank, nheads, temperature=1, device=None, dtype=None):
        """ Sets q = k
        Sets the columns of q to be an orthonormal basis for a random subspace of dimension rank
        Sets VO so to be identity, scaled so that the total function is an average over the heads
        """
        model = cls(dim=dim, rank=rank, nheads=nheads, device=device, dtype=dtype)
        # TODO should really make temperature infinity but whatever
        lr = torch.eye(rank, dim)
        for head in range(nheads):
            rotation, _ = torch.linalg.qr(torch.randn((dim, dim)))
            rotated_lr = lr @ rotation
            model.Q.data[head, :, :] = temperature * rotated_lr
            model.K.data[head, :, :] = rotated_lr
            model.VO.data[head, :, :] = torch.eye(dim, device=device, dtype=dtype) / nheads
        return model
    
    @classmethod
    def spaced_out_construction(cls, nheads, temperature=1, device=None, dtype=None):
        model = cls(dim=2, rank=1, nheads=nheads, device=device, dtype=dtype)
        angles = torch.linspace(0, torch.pi * (1 - 1/nheads), nheads, device=device, dtype=dtype)
        model.Q.data[:, 0, 0] = torch.cos(angles)
        model.Q.data[:, 0, 1] = torch.sin(angles)
        model.K.data = model.Q.data
        model.Q.data *= temperature
        for head in range(nheads):
            model.VO.data[head, :, :] = torch.eye(2, device=device, dtype=dtype) / nheads
        return model


class SoftMultiheadAttention(AbstractMultiheadAttention):
    def forward(self, X, Y):
        # X is batch_size, dim, num keys
        # self.Q and self.K are each nheads by rank by dim
        # attn_matrix is batch_size, num queries, num heads, num keys
        attn_matrix = torch.nn.Softmax(dim=3)(self.inside_heads(X, Y))
        return torch.einsum("bqhk,bdk,hde->beq", attn_matrix, X, self.VO)


class HardMultiheadAttention(AbstractMultiheadAttention):
    def forward(self, X, Y):
        # inside is batch_size, num queries, num heads, num keys
        # attended_to is batch_size, num queries, num heads
        attended_to = torch.argmax(self.inside_heads(X, Y), dim=3)
        # TODO: this is very wasteful. there should be a better way
        attn_matrix = torch.nn.functional.one_hot(attended_to, num_classes=X.shape[2]).float()
        # attn_matrix is batch_size, num queries, num heads, num points
        # X is batch_size, dim, num points
        return torch.einsum("bqhk,bdk,hde->beq", attn_matrix, X, self.VO)


class OptimallyWeightedRandom(HardMultiheadAttention):
    def __init__(self, dim, nheads, num_gegenbauer_terms, scipy_solver=False, seed=None, device=None, dtype=None):
        super().__init__(dim=dim, rank=1, nheads=nheads, device=device, dtype=dtype)
        rng = create_rng(seed, device)
        self.Q.data[:, 0, :] = samples_from_sphere(dim, nheads, rng).T
        self.K.data[:, 0, :] = samples_from_sphere(dim, nheads, rng).T

        if dim == 2:
            b = self.head_vs_target_2D(self.Q[:, 0, :].T, self.K[:, 0, :].T)
        else:
            b = self.approx_head_vs_target(self.Q[:, 0, :].T, self.K[:, 0, :].T, num_gegenbauer_terms)
        C = self.head_vs_head(self.Q[:, 0, :].T, self.K[:, 0, :].T)
        if scipy_solver:
            C = C.numpy(force=True)
            b = b.numpy(force=True)
            a = scipy.linalg.solve(C, b, assume_a='pos')
            a = torch.from_numpy(a).to(device)
        else:
            a = torch.linalg.solve(C, b)
        self.VO.data = torch.eye(dim, device=device, dtype=dtype).expand(nheads, dim, dim) * a.reshape(-1, 1, 1)

    @staticmethod
    def head_vs_head(Qs, Ks):
        def clip_asin(x):
            return torch.arcsin(torch.clip(x, -1, 1))

        return (1/2) + (2 / torch.pi**2) * clip_asin(Qs.T @ Qs) * clip_asin(Ks.T @ Ks)

    @staticmethod
    def head_vs_target_2D(Qs, Ks):
        angles = torch.arccos(torch.clip((Qs * Ks).sum(dim=0), -1, 1))
        fixed_angles = torch.pi/2 - torch.abs(torch.pi/2 - angles)
        return (1/2) + torch.sign(torch.pi/2 - angles) * (0.25 - (fixed_angles/torch.pi)**2)

    @staticmethod
    def approx_head_vs_target(Qs, Ks, num_gegenbauer_terms):
        # Qs and Ks have shape dim x num heads
        q_norms_inv = 1 / torch.norm(Qs, dim=0)
        k_norms_inv = 1 / torch.norm(Ks, dim=0)
        angles = torch.arccos(torch.clip((Qs * Ks).sum(axis=0) * q_norms_inv * k_norms_inv, -1, 1))
        angles = angles.numpy(force=True)
        a = HeadVsTarget(Qs.shape[0], num_gegenbauer_terms)(angles)
        return torch.from_numpy(a).float().to(Qs.device)


class CheatingWeights(AbstractMultiheadAttention):
    def __init__(self, dim, nheads, seed=None, device=None, dtype=None):
        super().__init__(dim=dim, rank=1, nheads=nheads, device=device, dtype=dtype)
        rng = create_rng(seed, device)
        self.Q.data[:, 0, :] = samples_from_sphere(dim, nheads, rng).T
        self.K.data[:, 0, :] = samples_from_sphere(dim, nheads, rng).T

        C = OptimallyWeightedRandom.head_vs_head(self.Q[:, 0, :].T, self.K[:, 0, :].T)
        del self.VO

    def forward(self, X, Y):
        # inside is batch_size, num queries, num heads, num keys
        # attended_to is batch_size, num queries, num heads
        attended_to = torch.argmax(self.inside_heads(X, Y), dim=3)
        # TODO: this is very wasteful. there should be a better way
        attn_matrix = torch.nn.functional.one_hot(attended_to, num_classes=X.shape[2]).float()
        # attn_matrix is batch_size, num queries, num heads, num points
        # X is batch_size, dim, num points
        features = torch.einsum("bqhk,bdk->bdqh", attn_matrix, X)
        # here's the cheating
        labels = self.label_batch(X, Y, X.device)
        # labels has dimensions: batch size, dim, num queries
        a = torch.linalg.lstsq(torch.flatten(features, end_dim=2), torch.flatten(labels)).solution
        return torch.einsum("bdqh,h->bdq", features, a)

    @staticmethod
    def label_batch(X, Y, device):
        # X is batch_size, dim, num points
        # Y is batch_size, dim, num queries
        labels = torch.empty(Y.shape, device=device)
        for ix in range(labels.shape[0]):
            labels[ix, :, :] = NearestPointDataset.label(X[ix, :, :], Y[ix, :, :])
        return labels
