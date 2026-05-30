"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from needle.backend_selection import default_device


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # TODO
        ### BEGIN YOUR SOLUTION
        device = default_device() if device is None else device
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = kernel_size * kernel_size * out_channels
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in,
                fan_out,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
                dtype=dtype,
            )
        )
        self.bias = (
            Parameter(init.zeros(out_channels, device=device, dtype=dtype))
            if bias
            else None
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor: # (N, C, H, W)
        # TODO
        ### BEGIN YOUR SOLUTION
        x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(
            x,
            self.weight,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        if self.bias is not None:
            bias = self.bias.reshape((1, 1, 1, self.out_channels))
            out = out + bias.broadcast_to(out.shape)
        return out.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION
