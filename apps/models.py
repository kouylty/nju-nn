import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
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
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x):
        # TODO
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
