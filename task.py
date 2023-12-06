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


class FarthestPointSentenceDataset(torch.utils.data.IterableDataset):
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

# Everything above this line is old

class FarthestPointSeparateDataset(torch.utils.data.IterableDataset):
    def __init__(self, dim, num_points, num_queries, double_points, seed=None, device=None):
        super().__init__()
        self.dim = dim
        self.num_points = num_points
        self.num_queries = num_queries
        self.double_points = double_points
        self.seed = seed
        self.device = device

    def create_rng(self):
        rng = torch.Generator(self.device)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        seed = rng.seed() if self.seed is None else self.seed
        rng.manual_seed(seed + worker_id)
        return rng

    @staticmethod
    def samples_from_sphere(dim, num_points, rng):
        X = torch.empty((dim, num_points), device=rng.device)
        torch.randn((dim, num_points), generator=rng, out=X)
        # normalize each row
        return X / X.norm(dim=0)

    @staticmethod
    def label(X, Y):
        return X[:, (X.T @ Y).argmax(axis=0)]

    def __iter__(self):
        rng = self.create_rng()
        while True:
            X = self.samples_from_sphere(self.dim, self.num_points, rng)
            if self.double_points:
                X = torch.concat((X, -X), dim=1)
            Y = self.samples_from_sphere(self.dim, self.num_queries, rng)
            yield X, Y, self.label(X, Y)
