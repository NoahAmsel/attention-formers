from math import sqrt

import lightning.pytorch as pl
import torch


class MergedLinearFormer(torch.nn.Module):
    def __init__(self, d_model, batch_first=True, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.QK = torch.nn.Parameter(torch.empty((self.d_model, self.d_model), device=device, dtype=dtype))
        self.VO = torch.nn.Parameter(torch.empty((self.d_model, self.d_model), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.QK)
        torch.nn.init.xavier_uniform_(self.VO)

    def perturb_parameters(self, scale=1.):
        self.QK.data += scale * torch.randn_like(self.QK.data)
        self.VO.data += scale * torch.randn_like(self.VO.data)

    def forward(self, x):
        # x is batch_size, ntokens, dim
        # attn_matrix is batch_size, ntokens, ntokens
        attn_matrix = torch.nn.Softmax(dim=2)(torch.einsum("btd,de,bue->btu", x, self.QK, x) / sqrt(self.d_model))
        return torch.einsum("btu,bud,de->bte", attn_matrix, x, self.VO)

    @classmethod
    def farthest_init(cls, d_model, batch_first=True, device=None, dtype=None):
        model = cls(d_model, batch_first=batch_first, device=device, dtype=dtype)
        model.QK.data = -10000 * torch.eye(model.d_model, device=device, dtype=dtype)
        model.VO.data = torch.eye(model.d_model, device=device, dtype=dtype)
        return model

    @classmethod
    def extract_attn(cls, my_layer):
        dim = my_layer.self_attn.out_proj.weight.data.shape[0]
        merged = cls(dim)
        merged.QK.data = my_layer.self_attn.in_proj_weight.data[:dim, :].T \
            @ my_layer.self_attn.in_proj_weight.data[dim:(2*dim), :]
        merged.VO.data = my_layer.self_attn.in_proj_weight.data[(
            2*dim):, :].T @ my_layer.self_attn.out_proj.weight.data.T @ my_layer.linear.weight.T
        return merged


class LitSequenceRegression(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("thing", self.lr_schedulers(), on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=.1, patience=10, threshold=.001, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            },
        }


if __name__ == "__main__":
    from task import FarthestPointDataset
    ntokens = 100
    dim = 100
    batch_size = 64
    lit_model = LitSequenceRegression(MergedLinearFormer(dim))
    # lit_model = LitSequenceRegression(MergedLinearFormer.farthest_init(dim))
    # lit_model.model.perturb_parameters(0.3)
    data = torch.utils.data.DataLoader(FarthestPointDataset(ntokens, dim), batch_size=batch_size, num_workers=0)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=200, callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')])

    trainer.fit(model=lit_model, train_dataloaders=data)
