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
