import lightning as L
import torch


def samples_from_sphere(dim, num_points, rng):
    X = torch.empty((dim, num_points), device=rng.device)
    torch.randn((dim, num_points), generator=rng, out=X)
    # normalize each row
    return X / X.norm(dim=0)


def create_rng(seed=None, device=None):
    rng = torch.Generator(device)
    worker_info = torch.utils.data.get_worker_info()
    worker_id = 0 if worker_info is None else worker_info.id
    seed = rng.seed() if seed is None else seed
    rng.manual_seed(seed + worker_id)
    return rng


class NearestPointDataset(torch.utils.data.IterableDataset):
    def __init__(self, dim, num_points, num_queries, seed=None, device=None):
        super().__init__()
        self.dim = dim
        self.num_points = num_points
        self.num_queries = num_queries
        self.seed = seed
        self.device = device

    def create_rng(self):
        return create_rng(seed=self.seed, device=self.device)

    @staticmethod
    def label_nearest(X, Y):
        return X[:, (X.T @ Y).argmax(axis=0)]

    @staticmethod
    def label_farthest(X, Y):
        return X[:, (X.T @ Y).argmin(axis=0)]
    
    def _draw_X(self, rng):
        return samples_from_sphere(self.dim, self.num_points, rng)

    def __iter__(self):
        rng = self.create_rng()
        while True:
            X = self._draw_X(rng)
            if self.num_queries == -1:  # TODO: make the special value None instead of -1?
                yield X, self.label_farthest(X, X)
            else:
                Y = samples_from_sphere(self.dim, self.num_queries, rng)
                yield X, Y, self.label_nearest(X, Y)


class NearestPointDatasetDoubled(NearestPointDataset):
    def _draw_X(self, rng):
        X = samples_from_sphere(self.dim, self.num_points, rng)
        X = torch.concat((X, -X), dim=1)
        return X


class NearestPointDatasetOrthogonal(NearestPointDataset):
    def _draw_X(self, rng):
        assert self.num_points <= self.dim
        rotation, _ = torch.linalg.qr(torch.randn((self.dim, self.dim), generator=rng, device=rng.device))
        return rotation[:, :self.num_points]


class NearestPointDataModule(L.LightningDataModule):
    def __init__(self, dataset_class: type[NearestPointDataset], dim: int, num_points: int, num_queries: int, batch_size: int, num_workers: int = 0, seed:int = None, device_name:str = None):
        super().__init__()
        self.save_hyperparameters()
        if device_name is None:
            self.device = None
        else:
            self.device = torch.device(device_name)

    @classmethod
    def from_name(cls, dataset_name: str, dim: int, num_points: int, num_queries: int, batch_size: int, num_workers: int = 0, seed: int = None, device_name: str = None):
        name2task = {"unif": NearestPointDataset, "ortho": NearestPointDatasetOrthogonal, "doubled": NearestPointDatasetDoubled}
        cls(
            dataset_class=name2task[dataset_name],
            dim=dim,
            num_points=num_points,
            num_queries=num_queries,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            device_name=device_name
        )

    def setup(self, stage):
        self.dataset = self.hparams.dataset_class(
            dim=self.hparams.dim,
            num_points=self.hparams.num_points,
            num_queries=self.hparams.num_queries,
            seed=self.hparams.seed,
            device=self.device
        )

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=((self.device is None) or (self.device.type == "cpu"))
        )

    def train_dataloader(self): return self.dataloader()

    def test_dataloader(self): return self.dataloader()

    def predict_dataloader(self): return self.dataloader()

    def state_dict(self):
        # TODO: does this work?
        return dict(dataset=self.dataset)

    def load_state_dict(self, state_dict):
        self.dataset = state_dict["dataset"]


def dataset(dataset_name: str, dim: int, num_points: int, num_queries: int, batch_size: int, num_workers: int = 0, seed: int = None, device_name: str = None):
    module = NearestPointDataModule.from_name(
        dataset_name=dataset_name,
        dim=dim,
        num_points=num_points,
        num_queries=num_queries,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        device_name=device_name
    )
    module.prepare_data()
    module.setup("fit")
    return module.dataloader()
