from fire import Fire
import lightning as pl
from omegaconf import OmegaConf as oc
import torch

from models import HardMultiheadAttention
from task import FarthestPointSeparateDataset


class Tester(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def test_step(self, batch, batch_ix):
        # x has dimensions: (batch size, dim, num points)
        # y has dimensions: (batch size, dim, num queries)
        # label has dimensions: (batch size, dim, num queries)
        x, y, label = batch
        _, dim, _ = label.shape
        label_hat = self.model(x, y)
        # MSE loss averages all the entry-wise errors
        # but we don't want to average over dimension of the vectors
        loss = torch.nn.functional.mse_loss(label_hat, label) * dim
        # TODO: what if we allow a sign error? choose the sign of each prediction to minimize error
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss


def test_model(model, config):
    lit_model = Tester(model)
    data = torch.utils.data.DataLoader(
        FarthestPointSeparateDataset(
            dim=config.dim,
            num_points=config.num_points,
            num_queries=config.num_queries,
            double_points=False,
        ),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    tester = pl.Trainer(limit_test_batches=config.num_batches, logger=False)
    tester.test(model=lit_model, dataloaders=data)


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
        num_batches=50,
        num_workers=4,
    )
    # Fire(test_random_construction)
