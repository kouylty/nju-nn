import math
from .init_basic import *
from ..backend_selection import default_device


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    # TODO
    ### BEGIN YOUR SOLUTION
    if kwargs.get("device", None) is None:
        kwargs["device"] = default_device()
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    shape = kwargs.pop("shape", (fan_in, fan_out))
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    # TODO
    ### BEGIN YOUR SOLUTION
    if kwargs.get("device", None) is None:
        kwargs["device"] = default_device()
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    shape = kwargs.pop("shape", (fan_in, fan_out))
    return randn(*shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # TODO
    ### BEGIN YOUR SOLUTION
    if kwargs.get("device", None) is None:
        kwargs["device"] = default_device()
    bound = math.sqrt(6.0 / fan_in)
    shape = shape if shape is not None else (fan_in, fan_out)
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    # TODO
    ### BEGIN YOUR SOLUTION
    if kwargs.get("device", None) is None:
        kwargs["device"] = default_device()
    std = math.sqrt(2.0 / fan_in)
    shape = shape if shape is not None else (fan_in, fan_out)
    return randn(*shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
