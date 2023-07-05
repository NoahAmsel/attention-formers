import torch


def gen_sentence(ntokens, dim, rng=None):
    x = torch.empty((ntokens, dim), device=rng.device)
    torch.randn((ntokens, dim), generator=rng, out=x)
    # normalize each row
    return x / x.norm(dim=1).reshape(-1, 1)


def label_farthest(sentence):
    distances = 2 - 2 * sentence @ sentence.T
    farthests = distances.argmax(dim=0)
    return sentence[farthests, :]


class FarthestPointDataset(torch.utils.data.IterableDataset):
    # TODO: let ntokens vary according to some reasonable distribution
    def __init__(self, ntokens, dim, seed=None, device=None):
        super().__init__()
        self.ntokens = ntokens
        self.dim = dim
        self.seed = seed
        self.device = device

    def __iter__(self):
        rng = torch.Generator(self.device)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        seed = rng.seed() if self.seed is None else self.seed
        rng.manual_seed(seed + worker_id)

        while True:
            sentence = gen_sentence(self.ntokens, self.dim, rng=rng)
            yield sentence, label_farthest(sentence)
