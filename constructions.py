"""For models that we construct instead of training, we can do away with
a lot of the infrastructure.
"""
from itertools import product
from math import pi

import torch
from tqdm import tqdm


class Rank1HardTention(torch.nn.Module):
    def __init__(self, dim, nheads, rank):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.rank = rank
        self.qs = torch.nn.Parameter(torch.empty(self.nheads, self.rank, self.dim), requires_grad=False)
        self.ks = torch.nn.Parameter(torch.empty(self.nheads, self.rank, self.dim), requires_grad=False)
        self.init()

    def init(self):
        self.qs.data = torch.randn_like(self.qs)
        self.qs.data /= self.qs.norm(dim=-1)[:, :, None]
        self.ks.data = torch.randn_like(self.ks)
        self.ks.data /= self.ks.norm(dim=-1)[:, :, None]

    def forward(self, x):
        """"""
        # the input x has dimensions batch_size, ntokens, dim
        # b: batch
        # s: source token
        # t: target token
        # d: ambient dimension (when calculating queries)
        # e: ambient dimension (when calculating keys)
        # h: head
        # r: rank
        attn_scores = torch.einsum("bsd,hrd,hre,bte->bhst", x, self.qs, self.ks, x)
        # attened_to has size (batch, head, source)
        attended_to = attn_scores.max(dim=-1).indices
        # TODO: oof this is wasteful. look into torch.gather
        # attn_matrices has size (batch, head, source, target="num_classes")
        attn_matrices = torch.nn.functional.one_hot(attended_to, num_classes=x.shape[1]).float()
        out = torch.einsum("bhst,bte->bse", attn_matrices, x)
        return out / out.norm(dim=2)[:, :, None]


class FullRankSolution(Rank1HardTention):
    def __init__(self, dim):
        super().__init__(dim=dim, nheads=1, rank=dim)

    def init(self):
        self.qs[0] = -torch.eye(self.dim)
        self.ks[0] = torch.eye(self.dim)


class QQTention(Rank1HardTention):
    def __init__(self, dim, nheads):
        super().__init__(dim=dim, nheads=nheads, rank=1)

    def init(self):
        self.qs.data = torch.randn_like(self.qs)
        self.qs.data /= self.qs.norm(dim=-1)[:, :, None]
        self.ks.data = -self.qs
    

def my_angular_error(model, batch):
    sentence, label = batch
    prediction = model(sentence)
    # dimensions of label and prediction are both (batch_size, ntokens, dim)
    normalizing_factors = prediction.norm(dim=-1) * label.norm(dim=-1)
    return torch.arccos(torch.clip(torch.einsum("btd,btd->bt", prediction, label) / normalizing_factors, min=-1, max=1)).mean()


def my_test(model, data_loader, num_batches, tqdm=False):
    batch_range = range(num_batches)
    if tqdm: batch_range = tqdm(batch_range)
    return sum(my_angular_error(model, next(iter(data_loader))) for _ in batch_range) / num_batches


if __name__ == "__main__":
    import pandas as pd
    from task import FarthestPointDataset

    rows = []
    # dim_set = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512]
    ntokens_set = [4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512]
    dim_set = [8]
    ntokens_set = [2**i for i in range(2, 14)]
    for dim, ntokens in tqdm(product(dim_set, ntokens_set)):
        nheads = 2 ** 9

        batch_size = 1
        num_batches = 64
        device = "cuda"
        data_loader = torch.utils.data.DataLoader(
            FarthestPointDataset(ntokens, dim, device=device),
            batch_size=batch_size,
            num_workers=0
        )

        # model = QQTention(dim, nheads).to(device)
        num_runs = 32
        error = sum(my_test(Rank1HardTention(dim, nheads, 1).to(device), data_loader, num_batches).item() for _ in range(num_runs)) / num_runs
        # error = sum(my_test(QQTention(dim, nheads).to(device), data_loader, num_batches).item() for _ in range(num_runs)) / num_runs
        rows.append(dict(
            dim=dim,
            ntokens=ntokens,
            error=error * 180/pi
        ))

    df = pd.DataFrame(rows)
    df.to_csv("results_qk_d16.csv", float_format='{:,.2f}'.format, index=False)