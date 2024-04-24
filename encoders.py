import torch

from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch import Tensor
from typing import Callable, Optional, Union


class MyNormalize(torch.nn.Module):
    def __init__(self, dim, scale=1, eps=0, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.scale = torch.nn.Parameter(torch.ones(1, device=device, dtype=dtype) * scale)
        self.eps = eps  # remove me?

    def forward(self, x):
        return self.scale * torch.nn.functional.normalize(x, dim=self.dim, eps=self.eps)


class WidenedTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, width_multiplier: int, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype
        )

        self.widened_self_attn = torch.nn.ParameterList([
            # Each entry in this list is constructed exactly as in TransformerEncoderLayer
            MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
                device=device,
                dtype=dtype
            )
            for _ in range(width_multiplier)
        ])

        self.norm1 = MyNormalize(dim=2)
        self.norm2 = MyNormalize(dim=2)

    # The TransformerEncoder initializer inspects properties of each layer's self_attn attribute.
    # Expose this so it can still do the job
    @property
    def self_attn(self):
        return self.widened_self_attn[0]

    # Exactly th e same as TransformerEncoderLayer.forward, but the fastpath block is removed
    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        out = torch.zeros_like(x)
        for sa in self.widened_self_attn:
            out = out + sa(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0] / len(self.widened_self_attn)
        return self.dropout1(out)


class Encoder(torch.nn.Module):
    def __init__(self, dim: int, nheads: int, dim_feedforward: int, num_layers: int, width_multiplier: int = 1, bias: bool = True, positional_dim: int = 0, maxN: int = 0):
        super().__init__()
        layer = WidenedTransformerEncoderLayer(width_multiplier=width_multiplier, d_model=(dim+positional_dim), nhead=nheads, dim_feedforward=dim_feedforward, dropout=0, batch_first=True, bias=bias)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.positional_encodings = torch.nn.Parameter(torch.empty(positional_dim, maxN))
        torch.nn.init.uniform_(self.positional_encodings, -1, 1)

    def positional_dim(self):
        return self.positional_encodings.shape[0]

    def maxN(self):
        return self.positional_encodings.shape[1]

    def forward(self, X):
        # X has dimensions: (batch size, dim, num points)
        batch_size, dim, num_points = X.shape
        if self.positional_dim() > 0:
            assert num_points <= self.maxN(), f"Number of points in batch {num_points} is larger than the Encoder can handle {self.maxN()}. Try increasing maxN."
            X = torch.cat([X, self.positional_encodings[:, :num_points].expand(batch_size, self.positional_dim(), num_points)], dim=1)
        # encoder layer input and output must have shape (batch size, num points, dim) because batch_first=True
        encoder_out = torch.permute(self.encoder(torch.permute(X, (0, 2, 1))), (0, 2, 1))
        # strip out the extra dimensions added by positional encoding
        return encoder_out[:, :dim, :]


class PerfectEncoder(Encoder):
    def __init__(self, dim: int, dim_feedforward: int, num_layers: int, width_multiplier: int = 1):
        super().__init__(dim=dim, nheads=1, dim_feedforward=dim_feedforward, num_layers=num_layers, width_multiplier=width_multiplier, bias=False, positional_dim=0)

        with torch.no_grad():
            # first layer should compute the function
            for attn in self.encoder.layers[0].widened_self_attn:
                attn.in_proj_weight[:dim, :] = -1e6 * torch.eye(dim)
                attn.in_proj_weight[dim:(2*dim), :] = torch.eye(dim)
                attn.in_proj_weight[(2*dim):, :] = torch.eye(dim)
                attn.out_proj.weight.data = 1e6 * torch.eye(dim)
            self.encoder.layers[0].linear1.weight.data = torch.zeros_like(self.encoder.layers[0].linear1.weight)
            self.encoder.layers[0].linear2.weight.data = torch.zeros_like(self.encoder.layers[0].linear2.weight)

            # subsequent layers just use the skip connections
            for layer in self.encoder.layers[1:]:
                for attn in layer.widened_self_attn:
                    attn.out_proj.weight.data = torch.zeros_like(attn.out_proj.weight.data)
                layer.linear1.weight.data = torch.zeros_like(layer.linear1.weight)
                layer.linear2.weight.data = torch.zeros_like(layer.linear2.weight)
