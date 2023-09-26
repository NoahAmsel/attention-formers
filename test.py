from fire import Fire
import lightning as pl
from omegaconf import OmegaConf as oc
import torch

from models import MultiheadSelfAttention
from task import FarthestPointDataset


class Tester(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def test_step(self, batch, batch_ix):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss


def test_random_construction(**config):
    config = oc.create(config)
    lit_model = Tester(MultiheadSelfAttention.random_construction(config.dim, config.rank, config.nheads))
    data = torch.utils.data.DataLoader(
        FarthestPointDataset(config.ntokens, config.dim),
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    tester = pl.Trainer(limit_test_batches=config.num_batches, logger=False)
    tester.test(model=lit_model, dataloaders=data)


if __name__ == "__main__":
    test_random_construction(
        dim=100,
        rank=100,
        nheads=1000,
        ntokens=100,
        batch_size=64,
        num_batches=100,
        num_workers=4,
    )
    # Fire(test_random_construction)
