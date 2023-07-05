"""For models that we construct instead of training, we can do away with
a lot of the infrastructure.
"""

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


def my_angular_error(model, batch):
    sentence, label = batch
    prediction = model(sentence)
    # dimensions of label and prediction are both (batch_size, ntokens, dim)
    normalizing_factors = prediction.norm(dim=-1) * label.norm(dim=-1)
    return torch.arccos(torch.clip(torch.einsum("btd,btd->bt", prediction, label) / normalizing_factors, min=-1, max=1)).mean()


def my_test(model, data_loader, num_batches):
    return sum(my_angular_error(model, next(iter(data_loader))) for _ in tqdm(range(num_batches))) / num_batches


if __name__ == "__main__":
    from task import FarthestPointDataset

    dim = 2**8
    ntokens = 2**8
    nheads = 2**8

    batch_size = 4
    num_batches = 8
    device = 'mps'
    data_loader = torch.utils.data.DataLoader(
        FarthestPointDataset(ntokens, dim, device=device),
        batch_size=batch_size,
        num_workers=0
    )

    model = FullRankSolution(dim)
    model = Rank1HardTention(dim, nheads, rank=5)
    model.to(device)

    print(my_test(model, data_loader, num_batches))
