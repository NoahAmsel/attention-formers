from fire import Fire
import lightning as pl
from omegaconf import OmegaConf as oc
import torch

from models import MultiheadSelfAttention, SeparateMultiheadSelfAttention
from task import FarthestPointDataset, FarthestPointSeparateDataset


class Tester(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def test_step(self, batch, batch_ix):
        x, y, label = batch
        label_hat = self.model(x, y)
        # MSE loss averages all the entry-wise errors
        # but we don't want to average over dimension of the vectors
        loss = torch.nn.functional.mse_loss(label_hat, label) * label.shape[-1]
        # # maybe the scaling is wrong. rescale. doesn't work
        # nbatches, ntokens, ndims = y.shape
        # scaled_y_hat = y_hat / torch.norm(y_hat, dim=2)[:, :, None]
        # loss = torch.norm(scaled_y_hat - y) ** 2 / (nbatches * ntokens)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss


def test_random_construction(**config):
    config = oc.create(config)
    lit_model = Tester(SeparateMultiheadSelfAttention.random_construction(config.dim, config.rank, config.nheads))
    data = torch.utils.data.DataLoader(
        FarthestPointSeparateDataset(config.ntokens, config.dim),
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    tester = pl.Trainer(limit_test_batches=config.num_batches, logger=False)
    tester.test(model=lit_model, dataloaders=data)


if __name__ == "__main__":
    test_random_construction(
        dim=100,
        rank=100,
        nheads=1,
        ntokens=1000,
        batch_size=256,
        num_batches=50,
        num_workers=4,
    )
    # Fire(test_random_construction)
