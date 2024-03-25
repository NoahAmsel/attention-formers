import git
import lightning as L
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import torch

from models import SoftMultiheadAttention
from task import NearestPointDataModule


class LitSequenceRegression(L.LightningModule):
    def __init__(self, model: torch.nn.Module, scale_batch: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

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
        if self.hparams.scale_batch:
            scale = (labels * labels_hat).sum() / (labels_hat ** 2).sum()
            labels_hat *= scale
        return torch.nn.functional.mse_loss(labels_hat, labels) * dim

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_ix):
        loss = self.loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    # def configure_optimizers(self):
    #     # TODO: SGD?
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     #     optimizer, factor=.1, patience=10, threshold=.01, verbose=True)
    #     # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     #     optimizer, T_max=self.hparams.epochs, verbose=True
    #     # )
    #     return {
    #         "optimizer": optimizer,
    #         # "lr_scheduler": {
    #         #     "scheduler": lr_scheduler,
    #         #     "monitor": "train_loss",
    #         # },
    #     }


class LitSoftmaxAttention(LitSequenceRegression):
    def __init__(self, dim: int, rank: int, nheads: int, scale_batch: bool = False):
        model = SoftMultiheadAttention(dim=dim, rank=rank, nheads=nheads)
        super().__init__(model=model, scale_batch=scale_batch)
        self.save_hyperparameters()
        # TODO: this doesn't seem to be working at all
        # TODO: should this be in LitSequenceRegression?
        self.hparams.git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha


class PerfectTraining(LitSoftmaxAttention):
    @staticmethod
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

    def on_train_epoch_end(self):
        QK_scale, QK_diag_error, QK_off_diag_error = self.compare_to_identity(self.model.assemble_QK(0))
        VO_scale, VO_diag_error, VO_off_diag_error = self.compare_to_identity(self.model.assemble_VO(0))
        self.log("QK_scale", QK_scale, logger=True)
        self.log("QK_diag_error", QK_diag_error, logger=True)
        self.log("QK_off_diag_error", QK_off_diag_error, logger=True)
        self.log("VO_scale", VO_scale, logger=True)
        self.log("VO_diag_error", VO_diag_error, logger=True)
        self.log("VO_off_diag_error", VO_off_diag_error, logger=True)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.dim", "model.dim")
        parser.add_optimizer_args(torch.optim.SGD)  # Adam
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

        parser.add_argument("--experiment_name", default="lightning_logs")
        # TODO: In future, instead of linking these deterministically, just
        # use variable interpolation in the default config file
        # this will also avoid ambiguity in the next line if there are more than one logger
        parser.link_arguments("experiment_name", "trainer.logger.init_args.name")

        # TODO: wandb logger, if some option is set
        # wandb_logger = WandbLogger(
        #     name=str(config.experiment_name),
        #     save_dir=config.wandb_log_parent_dir,
        #     version=f"{config.experiment_name}-{config.experiment_version}",
        #     project="attention-rank",
        #     entity="noahamselsteam",
        # )


def main(args: ArgsType = None):
    cli = MyLightningCLI(
        LitSoftmaxAttention,
        NearestPointDataModule,
        trainer_defaults=dict(
            callbacks=[
                # TODO: remove this restriction and alter limit_train_batches so that the total number of batches is constant
                # BatchSizeFinder(max_trials=9),
                # LearningRateFinder(),  # TODO: this doesn't seem to play nicely with configs. investigate
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(monitor="train_loss", save_last=True),
            ],
        ),
        args=args
    )


if __name__ == "__main__":
    main()
