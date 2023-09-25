from math import sqrt

import lightning.pytorch as pl
import torch


class LRSelfAttentionHead(torch.nn.Module):
    def __init__(self, dim, rank, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.Q = torch.nn.Parameter(torch.empty((rank, self.dim), device=device, dtype=dtype))
        self.K = torch.nn.Parameter(torch.empty((rank, self.dim), device=device, dtype=dtype))
        self.VO = torch.nn.Parameter(torch.empty((self.dim, self.dim), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Q)
        torch.nn.init.xavier_uniform_(self.K)
        torch.nn.init.xavier_uniform_(self.VO)        

    def assemble_QK(self):
        return self.Q.data.T @ self.K.data

    def assemble_VO(self):
        return self.VO.data

    def forward(self, x):
        # x is batch_size, ntokens, dim
        # self.Q and self.K are each rank by dim
        # attn_matrix is batch_size, ntokens, ntokens
        attn_matrix = torch.nn.Softmax(dim=2)(torch.einsum("btd,rd,re,bue->btu", x, self.Q, self.K, x) / sqrt(self.dim))
        return torch.einsum("btu,bud,de->bte", attn_matrix, x, self.VO)

    @classmethod
    def farthest_init(cls, dim, temperature=10000, device=None, dtype=None):
        model = cls(dim, dim, device=device, dtype=dtype)
        model.Q.data = -temperature * torch.eye(model.dim, device=device, dtype=dtype)
        model.K.data = torch.eye(model.dim, device=device, dtype=dtype)
        model.VO.data = torch.eye(model.dim, device=device, dtype=dtype)
        return model


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, dim, rank, nheads, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.Q = torch.nn.Parameter(torch.empty((nheads, rank, dim), device=device, dtype=dtype))
        self.K = torch.nn.Parameter(torch.empty((nheads, rank, dim), device=device, dtype=dtype))
        self.VO = torch.nn.Parameter(torch.empty((nheads, dim, dim), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Q)
        torch.nn.init.xavier_uniform_(self.K)
        torch.nn.init.xavier_uniform_(self.VO)        

    def assemble_QK(self, head_ix):
        return self.Q.data[head_ix, :, :].T @ self.K.data[head_ix, :, :]

    def assemble_VO(self, head_ix):
        return self.VO.data[head_ix, :, :]

    def forward(self, x):
        # x is batch_size, ntokens, dim
        # self.Q and self.K are each rank by dim
        # attn_matrix is batch_size, ntokens, ntokens
        attn_matrix = torch.nn.Softmax(dim=3)(torch.einsum("btd,hrd,hre,bue->bthu", x, self.Q, self.K, x) / sqrt(self.dim))
        return torch.einsum("bthu,bud,hde->bte", attn_matrix, x, self.VO)

    @classmethod
    def farthest_init(cls, dim, temperature=10000, device=None, dtype=None):
        model = cls(dim=dim, rank=dim, nheads=1, device=device, dtype=dtype)
        model.Q.data[0,:,:] = -temperature * torch.eye(model.dim, device=device, dtype=dtype)
        model.K.data[0,:,:] = torch.eye(model.dim, device=device, dtype=dtype)
        model.VO.data[0,:,:] = torch.eye(model.dim, device=device, dtype=dtype)
        return model


class MergedLinearFormer(torch.nn.Module):
    def __init__(self, dim, batch_first=True, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.QK = torch.nn.Parameter(torch.empty((self.dim, self.dim), device=device, dtype=dtype))
        self.VO = torch.nn.Parameter(torch.empty((self.dim, self.dim), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.QK)
        torch.nn.init.xavier_uniform_(self.VO)

    # def perturb_parameters(self, scale=1.):
    #     self.QK.data += scale * torch.randn_like(self.QK.data)
    #     self.VO.data += scale * torch.randn_like(self.VO.data)

    def assemble_QK(self):
        return self.QK.data

    def assemble_VO(self):
        return self.VO.data

    def forward(self, x):
        # x is batch_size, ntokens, dim
        # attn_matrix is batch_size, ntokens, ntokens
        attn_matrix = torch.nn.Softmax(dim=2)(torch.einsum("btd,de,bue->btu", x, self.QK, x) / sqrt(self.dim))
        return torch.einsum("btu,bud,de->bte", attn_matrix, x, self.VO)

    @classmethod
    def farthest_init(cls, d_model, batch_first=True, device=None, dtype=None):
        model = cls(d_model, batch_first=batch_first, device=device, dtype=dtype)
        model.QK.data = -10000 * torch.eye(model.dim, device=device, dtype=dtype)
        model.VO.data = torch.eye(model.dim, device=device, dtype=dtype)
        return model
