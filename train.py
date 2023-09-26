import git
import lightning as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf as oc
import torch

from models import MultiheadSelfAttention
from task import FarthestPointDataset


def compare_to_identity(matrix):
    dim = matrix.shape[0]
    assert dim == matrix.shape[1]
    # normalize to scale of I. l1 norm of diagonal should be dim to match that of I
    scale = torch.norm(matrix.diagonal(), p=1) / dim
    scaled_matrix = matrix / scale
    max_diag_error = torch.norm(scaled_matrix.diagonal() - 1, p=torch.inf)
    scaled_matrix.fill_diagonal_(0)
    max_off_diag_error = torch.norm(scaled_matrix, p=torch.inf)
    return scale, max_diag_error, max_off_diag_error


class LitSequenceRegression(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiheadSelfAttention(self.config.dim, self.config.rank, self.config.nheads)

    @property
    def config(self):
        return self.hparams

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # QK should be like -I, not I. Hence the minus sign
        QK_scale, QK_diag_error, QK_off_diag_error = compare_to_identity(-self.model.assemble_QK(0))
        VO_scale, VO_diag_error, VO_off_diag_error = compare_to_identity(self.model.assemble_VO(0))
        self.log("QK_scale", QK_scale, logger=True)
        self.log("QK_diag_error", QK_diag_error, logger=True)
        self.log("QK_off_diag_error", QK_off_diag_error, logger=True)
        self.log("VO_scale", VO_scale, logger=True)
        self.log("VO_diag_error", VO_diag_error, logger=True)
        self.log("VO_off_diag_error", VO_off_diag_error, logger=True)

    def test_step(self, batch, batch_ix):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=.1, patience=10, threshold=.01, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            },
        }


def train(config):
    # Update git hash with current commit
    repo = git.Repo(config.code_dir, search_parent_directories=True)
    config.git_hash = repo.head.object.hexsha

    data = torch.utils.data.DataLoader(
        FarthestPointDataset(config.ntokens, config.dim),
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    lit_model = LitSequenceRegression(**config)

    csv_logger = CSVLogger(
        save_dir=config.csv_log_dir,
        name=config.experiment_name,
        version=config.experiment_version
    )
    if config.skip_wandb:
        logger = [csv_logger]
    else:
        wandb_logger = WandbLogger(
            name=str(config.experiment_name),
            save_dir=config.wandb_log_parent_dir,
            version=f"{config.experiment_name}-{config.experiment_version}",
            project="attention-rank",
            entity="noahamselsteam",
        )
        logger = [csv_logger, wandb_logger]

    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[pl.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')]
    )
    # if not config.skip_wandb:
    #     wandb_logger.watch(lit_model, log="all", log_freq=10)
    trainer.fit(model=lit_model, train_dataloaders=data)
    # if not config.skip_wandb:
    #     wandb_logger.experiment.unwatch(lit_model)


if __name__ == "__main__":
    train(oc.create(dict(
        dim=100,
        ntokens=100,
        rank=100,
        nheads=1,
        lr=1e-2,
        batch_size=64,
        experiment_name="default_experiment",
        experiment_version=None,
        code_dir="/home/nia4240/attention-formers",
        csv_log_dir="/home/nia4240/attention-formers/csv_logs",
        wandb_log_parent_dir="/home/nia4240/attention-formers/wandb_logs",
        skip_wandb=True,
        debug=True,
    )))
