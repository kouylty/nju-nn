from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 


def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(axis if axis >= 0 else axis + ndim for axis in axes)


def _keepdims_shape(shape, axes):
    axes = set(axes)
    return tuple(1 if i in axes else dim for i, dim in enumerate(shape))


def _remove_axes_shape(shape, axes):
    axes = set(axes)
    return tuple(dim for i, dim in enumerate(shape) if i not in axes)


def _reduce_keepdims(a, axes, reduce_fn):
    out = a
    for axis in sorted(axes, reverse=True):
        out = reduce_fn(out, axis=axis)
        shape = list(out.shape)
        shape.insert(axis, 1)
        out = array_api.reshape(out, tuple(shape))
    return out


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        axes = (len(Z.shape) - 1,)
        Z_max = _reduce_keepdims(Z, axes, array_api.max)
        shifted = Z - array_api.broadcast_to(Z_max, Z.shape)
        Z_sum = _reduce_keepdims(array_api.exp(shifted), axes, array_api.sum)
        log_sum = array_api.log(Z_sum)
        return shifted - array_api.broadcast_to(log_sum, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        axis = len(Z.shape) - 1
        keep_shape = _keepdims_shape(Z.shape, (axis,))
        lse = logsumexp(Z, axes=axis).reshape(keep_shape)
        probs = exp(Z - broadcast_to(lse, Z.shape))
        grad_sum = summation(out_grad, axes=axis).reshape(keep_shape)
        return out_grad - probs * broadcast_to(grad_sum, Z.shape)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        axes = _normalize_axes(self.axes, len(Z.shape))
        Z_max = _reduce_keepdims(Z, axes, array_api.max)
        shifted = Z - array_api.broadcast_to(Z_max, Z.shape)
        Z_sum = _reduce_keepdims(array_api.exp(shifted), axes, array_api.sum)
        out = array_api.log(Z_sum) + Z_max
        return array_api.reshape(out, _remove_axes_shape(Z.shape, axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        axes = _normalize_axes(self.axes, len(Z.shape))
        keep_shape = _keepdims_shape(Z.shape, axes)
        lse = logsumexp(Z, axes=self.axes).reshape(keep_shape)
        grad = out_grad.reshape(keep_shape)
        return broadcast_to(grad, Z.shape) * exp(Z - broadcast_to(lse, Z.shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
