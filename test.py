import torch
import lightning as L

from models import HardMultiheadAttention, OptimallyWeightedRandom
from task import dataset
from train import LitSequenceRegression


def test_model(model, num_test_batches, scale_batch=False, **dataset_kwargs):
    data = dataset(**dataset_kwargs)
    lit_model = LitSequenceRegression(model, scale_batch=scale_batch)
    tester = L.Trainer(limit_test_batches=num_test_batches, logger=False)
    return tester.test(model=lit_model, dataloaders=data)


class ZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Lighting will automatically move this parameter to the right device
        # Then when forward creates a tensor, it knows which device to put it on
        self.device_dummy = torch.nn.Parameter(torch.empty(0))

    def forward(self, X, Y):
        return torch.zeros(Y.shape, device=self.device_dummy.device)


def test_zero_construction(**config):
    """Useful for testing that the loss is calculated correctly.
    Loss should always be 1.
    """
    test_model(ZeroModel(), **config)


def test_perfect_construction(**config):
    model = HardMultiheadAttention.perfect_construction(dim=config.dim)
    test_model(model, **config)


def test_random_construction(**config):
    model = HardMultiheadAttention.random_construction(
        config["dim"], config["rank"], config["nheads"]
    )
    test_model(model, **config)


def test_spaced_construction(**config):
    assert config["dim"] == 2
    assert config["rank"] == 1
    model = HardMultiheadAttention.spaced_out_construction(config["nheads"])
    test_model(model, **config)


def test_optimally_weighted(**config):
    assert config["rank"] == 1
    model = OptimallyWeightedRandom(config["dim"], config["nheads"], config["num_gegenbauer_terms"])
    test_model(model, **config)


if __name__ == "__main__":
    test_optimally_weighted(
        dim=2,
        rank=1,
        nheads=2**10,
        num_points=2,
        num_queries=4,
        dataset_name="ortho",
        scale_batch=False,
        batch_size=128,
        num_test_batches=256,
        num_workers=4,
        num_gegenbauer_terms=100,
    )
    # Fire(test_random_construction)
    # TODO: make separate command line endpoints for test random, test perfect, test zero, etc
