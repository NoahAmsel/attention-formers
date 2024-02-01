from abc import ABC, abstractmethod
from math import sqrt

import torch


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
        model.Q.data[0,:,:] = temperature * torch.eye(model.dim, device=device, dtype=dtype)
        model.K.data[0,:,:] = torch.eye(model.dim, device=device, dtype=dtype)
        model.VO.data[0,:,:] = torch.eye(model.dim, device=device, dtype=dtype)
        return model

    @classmethod
    def random_construction(cls, dim, rank, nheads, temperature=1, device=None, dtype=None):
        model = cls(dim=dim, rank=rank, nheads=nheads, device=device, dtype=dtype)
        # TODO should really make temperature infinity but whatever
        lr = torch.eye(rank, dim)
        for head in range(nheads):
            rotation, _ = torch.linalg.qr(torch.randn((dim, dim)))
            rotated_lr = lr @ rotation
            model.Q.data[head,:,:] = temperature * rotated_lr
            model.K.data[head,:,:] = rotated_lr
            model.VO.data[head,:,:] = torch.eye(dim, device=device, dtype=dtype) / nheads
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
