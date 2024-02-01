import git
import lightning as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf as oc
import torch

from models import SoftMultiheadAttention, HardMultiheadAttention
from task import FarthestPointSeparateDataset


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
    def __init__(self, model, **config):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    @property
    def config(self):
        return self.hparams

    def loss(self, batch):
        # X has dimensions: (batch size, dim, num points)
        # Y has dimensions: (batch size, dim, num queries)
        # labels has dimensions: (batch size, dim, num queries)
        X, Y, labels = batch
        _, dim, _ = labels.shape
        labels_hat = self.model(X, Y)
        # MSE loss averages all the entry-wise errors
        # but we don't want to average over dimension of the vectors,
        # so mulitply by dim
        return torch.nn.functional.mse_loss(labels_hat, labels) * dim

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_ix):
        loss = self.loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=.1, patience=10, threshold=.01, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            },
        }


class PerfectTraining(LitSequenceRegression):
    def on_train_epoch_end(self):
        QK_scale, QK_diag_error, QK_off_diag_error = compare_to_identity(self.model.assemble_QK(0))
        VO_scale, VO_diag_error, VO_off_diag_error = compare_to_identity(self.model.assemble_VO(0))
        self.log("QK_scale", QK_scale, logger=True)
        self.log("QK_diag_error", QK_diag_error, logger=True)
        self.log("QK_off_diag_error", QK_off_diag_error, logger=True)
        self.log("VO_scale", VO_scale, logger=True)
        self.log("VO_diag_error", VO_diag_error, logger=True)
        self.log("VO_off_diag_error", VO_off_diag_error, logger=True)


def dataset(config):
    return torch.utils.data.DataLoader(
        FarthestPointSeparateDataset(
            dim=config.dim,
            num_points=config.num_points,
            num_queries=config.num_queries,
            double_points=False
        ),
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )


def train(config):
    # Update git hash with current commit
    repo = git.Repo(config.code_dir, search_parent_directories=True)
    config.git_hash = repo.head.object.hexsha

    data = dataset(**config)
    model = SoftMultiheadAttention(config.dim, config.rank, config.nheads)
    lit_model = LitSequenceRegression(model, **config)

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


def test(model, config):
    data = dataset(config)
    lit_model = LitSequenceRegression(model, **config)
    tester = pl.Trainer(limit_test_batches=config.num_batches, logger=False)
    tester.test(model=lit_model, dataloaders=data)


if __name__ == "__main__":
    train(oc.create(dict(
        dim=100,
        ntokens=100,
        rank=100,
        nheads=1,
        batch_size=64,
        lr=1e-2,
        epochs=250,
        num_workers=4,
        experiment_name="default_experiment",
        experiment_version=None,
        skip_wandb=True,
        debug=True,
        code_dir="/home/nia4240/attention-formers",
        csv_log_dir="/home/nia4240/attention-formers/csv_logs",
        wandb_log_parent_dir="/home/nia4240/attention-formers/wandb_logs",
    )))
