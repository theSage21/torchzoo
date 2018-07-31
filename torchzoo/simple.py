import torch.nn as nn


class FeedForward(nn.Module):
    """
    Simple Feed forward network.
    """
    def __init__(self, cell, input_dim, layer_dims):
        """
        For solving the XOR problem in two bits one would:

            FeedForward(2, [3, 3, 1])
        """
        super().__init__()
        self.cell = cell
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.layers = []
        idim = input_dim
        for dim in layer_dims:
            self.layers.append(self.cell(idim, dim))
            idim = dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class COOL(nn.Module):
    """
    Competitive Overcomplete Output Layer
    https://arxiv.org/pdf/1609.02226.pdf
    """
    def __init__(self, inp_dim, n_classes, doo):
        super().__init__()
        self.layer = nn.Linear(inp_dim, n_classes * doo)

    def forward(self, x):
        return nn.Softmax(dim=-1)(self.layer(x))
