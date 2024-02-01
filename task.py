import torch


class NearestPointDataset(torch.utils.data.IterableDataset):
    def __init__(self, dim, num_points, num_queries, seed=None, device=None):
        super().__init__()
        self.dim = dim
        self.num_points = num_points
        self.num_queries = num_queries
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

    def _draw_X(self, rng):
        return self.samples_from_sphere(self.dim, self.num_points, rng)

    def __iter__(self):
        rng = self.create_rng()
        while True:
            X = self._draw_X(rng)
            Y = self.samples_from_sphere(self.dim, self.num_queries, rng)
            yield X, Y, self.label(X, Y)


class NearestPointDatasetDoubled(NearestPointDataset):
    def _draw_X(self, rng):
        X = self.samples_from_sphere(self.dim, self.num_points, rng)
        X = torch.concat((X, -X), dim=1)
        return X


class NearestPointDatasetOrthogonal(NearestPointDataset):
    def _draw_X(self, rng):
        assert self.num_points <= self.dim
        rotation, _ = torch.linalg.qr(torch.randn((self.dim, self.dim), generator=rng, device=rng.device))
        return rotation[:, :self.num_points]
