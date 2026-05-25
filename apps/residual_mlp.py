import sys
from functools import reduce

sys.path.append("./python")

import needle as ndl
import needle.nn as nn


def _prod(shape):
    return reduce(lambda a, b: a * b, shape, 1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, device=None, dtype="float32"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Residual(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
                )
            ),
            nn.ReLU(),
        )

    def forward(self, x: ndl.Tensor) -> ndl.Tensor:
        return self.block(x)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_blocks=2,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks

        layers = [
            nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualMLPBlock(hidden_dim, device=device, dtype=dtype))
        layers.append(nn.Linear(hidden_dim, num_classes, device=device, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, x: ndl.Tensor) -> ndl.Tensor:
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], _prod(x.shape[1:])))
        return self.net(x)
