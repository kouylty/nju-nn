import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        super().__init__()
        self.conv2d = nn.Conv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            device=device, 
            dtype=dtype,
        )
        self.bn = nn.BatchNorm2d(
            dim=out_channels,
            device=device, 
            dtype=dtype,
        )
        self.relu = nn.ReLU()

    def forward(self, x: ndl.Tensor):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        # TODO
        ### BEGIN YOUR SOLUTION ###
        self.net = nn.Sequential(
            ConvBatchNorm(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBatchNorm(16, 32, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
                    ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
                )
            ),
            ConvBatchNorm(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBatchNorm(64, 128, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
                    ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        # TODO
        ### BEGIN YOUR SOLUTION
        return self.net(x)
        ### END YOUR SOLUTION
