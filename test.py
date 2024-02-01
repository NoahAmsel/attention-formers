from fire import Fire
from omegaconf import OmegaConf as oc
import torch

from models import HardMultiheadAttention
from train import test_model


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
    config = oc.create(config)
    test_model(ZeroModel(), config)


def test_perfect_construction(**config):
    config = oc.create(config)
    model = HardMultiheadAttention.perfect_construction(dim=config.dim)
    test_model(model, config)


def test_random_construction(**config):
    config = oc.create(config)
    model = HardMultiheadAttention.random_construction(
        config.dim, config.rank, config.nheads
    )
    test_model(model, config)


if __name__ == "__main__":
    test_random_construction(
        dim=20,
        rank=1,
        nheads=123,
        num_points=30,
        num_queries=5,
        batch_size=16,
        num_test_batches=50,
        num_workers=4,
    )
    # Fire(test_random_construction)
    # TODO: make separate command line endpoints for test random, test perfect, test zero, etc
