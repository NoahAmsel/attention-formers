from math import sqrt
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch


class LRSelfAttentionHead(torch.nn.Module):
    def __init__(self, dim, rank, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.Q = torch.nn.Parameter(torch.empty((rank, self.dim), device=device, dtype=dtype))
        self.K = torch.nn.Parameter(torch.empty((rank, self.dim), device=device, dtype=dtype))
        self.VO = torch.nn.Parameter(torch.empty((self.dim, self.dim), device=device, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Q)
        torch.nn.init.xavier_uniform_(self.K)
        torch.nn.init.xavier_uniform_(self.VO)        

    def assemble_QK(self):
        return self.Q.data.T @ self.K.data

    def assemble_VO(self):
        return self.VO.data

    def forward(self, x):
        # x is batch_size, ntokens, dim
        # self.Q and self.K are each rank by dim
        # attn_matrix is batch_size, ntokens, ntokens
        attn_matrix = torch.nn.Softmax(dim=2)(torch.einsum("btd,rd,re,bue->btu", x, self.Q, self.K, x) / sqrt(self.dim))
        return torch.einsum("btu,bud,de->bte", attn_matrix, x, self.VO)

    @classmethod
    def farthest_init(cls, dim, device=None, dtype=None):
        model = cls(dim, dim, device=device, dtype=dtype)
        model.Q.data = -10000 * torch.eye(model.dim, device=device, dtype=dtype)
        model.K.data = torch.eye(model.dim, device=device, dtype=dtype)
        model.VO.data = torch.eye(model.dim, device=device, dtype=dtype)
        return model


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

    def assemble_QK(self):
        return self.QK.data

    def assemble_VO(self):
        return self.VO.data

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

def angle_deg(a, b):
    return torch.acos(torch.inner(a, b) / (torch.norm(a) * torch.norm(b))) * 180 / torch.pi

class LitSequenceRegression(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("angle", (180 / torch.pi) * torch.acos(1 - loss/2), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        dim = self.model.assemble_QK().shape[0]
        # normalize to scale of I. l1 norm of diagonal should be dim to match I
        QK = self.model.assemble_QK()
        QK_scale = torch.norm(QK.diagonal(), p=1) / dim
        scaled_QK = QK / QK_scale
        VO = self.model.assemble_VO()
        VO_scale = torch.norm(VO.diagonal(), p=1) / dim
        scaled_VO = VO / VO_scale
        self.log("QK error", torch.norm(scaled_QK + torch.eye(dim, device=scaled_QK.device), p=torch.inf), prog_bar=True, logger=True)
        self.log("QK scale", QK_scale, prog_bar=True, logger=True)
        self.log("VO scale", VO_scale, prog_bar=True, logger=True)
        self.log("VO error", torch.norm(scaled_VO - torch.eye(dim, device=scaled_VO.device), p=torch.inf), prog_bar=True, logger=True)

    def test_step(self, batch, batch_ix):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
    lit_model = LitSequenceRegression(MergedLinearFormer(dim), lr=1e-1)
    lit_model = LitSequenceRegression(LRSelfAttentionHead(dim, int(dim * .9)), lr=1e-2)
    # lit_model = LitSequenceRegression(MergedLinearFormer.farthest_init(dim))
    # lit_model.model.perturb_parameters(0.3)
    data = torch.utils.data.DataLoader(FarthestPointDataset(ntokens, dim), batch_size=batch_size, num_workers=4)
    trainer = pl.Trainer(limit_train_batches=100, limit_test_batches=100, max_epochs=200, callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')])
    trainer.fit(model=lit_model, train_dataloaders=data)
    trainer.test(model=lit_model, dataloaders=data)
