"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
from functools import reduce
import needle.init as init
import numpy as np
from needle.backend_selection import default_device


def _prod(shape):
    return reduce(lambda a, b: a * b, shape, 1)


def _sum_keepdims(x: Tensor, axes):
    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(axis if axis >= 0 else axis + len(x.shape) for axis in axes)
    out = x.sum(axes=axes)
    shape = list(x.shape)
    for axis in axes:
        shape[axis] = 1
    return out.reshape(tuple(shape))


def _mean_keepdims(x: Tensor, axes):
    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(axis if axis >= 0 else axis + len(x.shape) for axis in axes)
    denom = _prod([x.shape[axis] for axis in axes])
    return _sum_keepdims(x, axes) / denom


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.kaiming_uniform(
            fan_in=in_features, 
            fan_out=out_features, 
            device=device, 
            dtype=dtype,
          )
        )
        self.bias = Parameter(
          init.kaiming_uniform(
            fan_in=out_features,
            fan_out=1,
            device=device, 
            dtype=dtype,
          ).reshape((1, out_features))
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight # (n, out_features)

        if self.bias is not None:
          y += ops.broadcast_to(self.bias, (*X.shape[:-1], self.out_features))
        
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # TODO
        ### BEGIN YOUR SOLUTION
        flattened_dim = reduce(lambda a, b: a * b, X.shape[1:])
        return X.reshape((X.shape[0], flattened_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        exp_neg = ops.exp(-x)
        return (exp_neg + 1) ** -1


class Sequential(Module):
    def __init__(self, *modules: List["Module"]):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.exp(ops.logsoftmax(x))


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # TODO
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        y_one_hot = init.one_hot(
            num_classes,
            y,
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = ops.logsumexp(logits, axes=1) - (logits * y_one_hot).sum(axes=1)
        return loss.sum() / batch_size
        ### END YOUR SOLUTION
        
class CrossEntrophyLoss(Module):
    def forward(self,logits: Tensor, y: Tensor):
        # TODO
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0]
        log_probs = ops.logsoftmax(logits)
        return -(y * log_probs).sum() / batch_size
        ### END YOUR SOLUTION
        
class BinaryCrossEntrophyLoss(Module):
    def forward(self,logits: Tensor, y: Tensor):
        # TODO
        ### BEGIN YOUR SOLUTION
        eps = 1e-7
        probs = logits
        loss = -(y * ops.log(probs + eps) + (-y + 1) * ops.log(-probs + 1 + eps))
        return loss.sum() / _prod(loss.shape)
        ### END YOUR SOLUTION
        
class MSELoss(Module):
    def forward(self, input: Tensor, target: Tensor):
        # TODO
        ### BEGIN YOUR SOLUTION
        loss = (input - target) ** 2
        return loss.sum() / _prod(loss.shape)
        ### END YOUR SOLUTION
        
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # TODO
        ### BEGIN YOUR SOLUTION
        device = default_device() if device is None else device
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 2 and x.shape[1] == self.dim
        batch_size = x.shape[0]
        stat_shape = (1, self.dim)

        if self.training:
            mean = x.sum(axes=0) / batch_size
            mean_b = mean.reshape(stat_shape).broadcast_to(x.shape)
            var = ((x - mean_b) ** 2).sum(axes=0) / batch_size

            self.running_mean.data = (
                self.running_mean.data * (1 - self.momentum)
                + mean.detach() * self.momentum
            )
            self.running_var.data = (
                self.running_var.data * (1 - self.momentum)
                + var.detach() * self.momentum
            )
        else:
            mean = self.running_mean
            var = self.running_var
            mean_b = mean.reshape(stat_shape).broadcast_to(x.shape)

        var_b = var.reshape(stat_shape).broadcast_to(x.shape)
        weight_b = self.weight.reshape(stat_shape).broadcast_to(x.shape)
        bias_b = self.bias.reshape(stat_shape).broadcast_to(x.shape)
        return ((x - mean_b) / ((var_b + self.eps) ** 0.5)) * weight_b + bias_b
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # TODO
        ### BEGIN YOUR SOLUTION
        device = default_device() if device is None else device
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        assert x.shape[-1] == self.dim
        axes = (len(x.shape) - 1,)
        mean = _mean_keepdims(x, axes)
        mean_b = mean.broadcast_to(x.shape)
        var = _mean_keepdims((x - mean_b) ** 2, axes)
        var_b = var.broadcast_to(x.shape)
        norm = (x - mean_b) / ((var_b + self.eps) ** 0.5)
        affine_shape = (1,) * (len(x.shape) - 1) + (self.dim,)
        weight = self.weight.reshape(affine_shape).broadcast_to(x.shape)
        bias = self.bias.reshape(affine_shape).broadcast_to(x.shape)
        return norm * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        if not self.training or self.p == 0:
            return x
        if self.p >= 1:
            return init.zeros_like(x)
        mask = init.randb(
            *x.shape,
            p=1 - self.p,
            device=x.device,
            dtype=x.dtype,
        )
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


class CrossEntropyLoss(CrossEntrophyLoss):
    pass


class BinaryCrossEntropyLoss(BinaryCrossEntrophyLoss):
    pass
