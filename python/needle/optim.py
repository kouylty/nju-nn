"""Optimization module"""
import needle as ndl
import numpy as np
import math


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        # TODO
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.detach()
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            if self.momentum != 0:
                if param not in self.u:
                    self.u[param] = grad
                else:
                    self.u[param] = self.momentum * self.u[param] + grad
                self.u[param] = self.u[param].detach()
                update = self.u[param]
            else:
                update = grad

            param.data = (param.data - self.lr * update).detach()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        # TODO
        ### BEGIN YOUR SOLUTION
        total_norm_sq = 0.0
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total_norm_sq += float((grad * grad).sum().numpy())

        total_norm = np.sqrt(total_norm_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for param in self.params:
                if param.grad is not None:
                    param.grad = (param.grad.detach() * scale).detach()
        return total_norm
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        # TODO
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.detach()
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            if param not in self.u:
                self.u[param] = ndl.init.zeros_like(param)
                self.v[param] = ndl.init.zeros_like(param)

            self.u[param] = self.beta1 * self.u[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad * grad)
            self.u[param] = self.u[param].detach()
            self.v[param] = self.v[param].detach()

            u_hat = self.u[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            update = (u_hat / ((v_hat ** 0.5) + self.eps)).detach()
            param.data = (param.data - self.lr * update).detach()
        ### END YOUR SOLUTION


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = -1

    def get_lr(self):
        raise NotImplementedError()

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self.optimizer.lr = lr
        return lr


class StepDecay(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class LinearWarmUp(LRScheduler):
    def __init__(self, optimizer, warmup_steps, start_lr=0.0):
        super().__init__(optimizer)
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if start_lr < 0:
            raise ValueError("start_lr must be non-negative")
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr

    def get_lr(self):
        if self.last_epoch >= self.warmup_steps:
            return self.base_lr
        ratio = (self.last_epoch + 1) / self.warmup_steps
        return self.start_lr + (self.base_lr - self.start_lr) * ratio


class CosineDecayWithWarmRestarts(LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, min_lr=0.0, cycle_mult=1.0):
        super().__init__(optimizer)
        if first_cycle_steps <= 0:
            raise ValueError("first_cycle_steps must be positive")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if cycle_mult < 1:
            raise ValueError("cycle_mult must be at least 1")
        self.first_cycle_steps = first_cycle_steps
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult

    def _cycle_position(self):
        step = self.last_epoch
        cycle_steps = self.first_cycle_steps
        while step >= cycle_steps:
            step -= cycle_steps
            cycle_steps = int(cycle_steps * self.cycle_mult)
        return step, cycle_steps

    def get_lr(self):
        cycle_step, cycle_steps = self._cycle_position()
        cosine = (1 + math.cos(math.pi * cycle_step / cycle_steps)) / 2
        return self.min_lr + (self.base_lr - self.min_lr) * cosine
