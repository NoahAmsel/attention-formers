import torch


class Encoder(torch.nn.Module):
    def __init__(self, dim: int, nheads: int, dim_feedforward: int, num_layers: int, width_multiplier: int = 1, bias: bool = True, positional_dim: int = 0, maxN: int = 0):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(d_model=(dim+positional_dim), nhead=nheads, dim_feedforward=dim_feedforward, dropout=0, batch_first=True, bias=bias)
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
        assert num_points <= self.maxN(), f"Number of points in batch {num_points} is larger than the Encoder can handle {self.maxN()}. Try increasing maxN."
        if self.positional_dim() > 0:
            X = torch.cat([X, self.positional_encodings[:, :num_points].expand(batch_size, self.positional_dim(), num_points)], dim=1)
        # encoder layer input and output must have shape (batch size, num points, dim) because batch_first=True
        encoder_out = torch.permute(self.encoder(torch.permute(X, (0, 2, 1))), (0, 2, 1))
        # strip out the extra dimensions added by positional encoding
        return encoder_out[:, :dim, :]
