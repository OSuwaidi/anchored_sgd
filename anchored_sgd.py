# بسم الله الرحمن الرحيم وبه نستعين

import torch
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    r"""Implements Vanilla Stochastic Gradient Descent (SGD) optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-1).
        beta (float, optional): Momentum factor (default: 0.0).
        dampening (float, optional): Dampening for momentum (default: 0.0).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        nesterov (bool, optional): Enables Nesterov momentum (default: False).
    """

    def __init__(
            self,
            params,
            lr: float = 1e-1,
            beta: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
            ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid momentum value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
        if nesterov and (beta <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=beta,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            )
        super().__init__(params, defaults)  # exposes "self.param_groups"

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            The loss returned by the closure, if any.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay:
                    grad = grad.add(p, alpha=weight_decay)

                if beta:
                    param_state = self.state[p]
                    if "momentum" not in param_state:
                        param_state["momentum"] = torch.zeros_like(grad)

                    m = param_state["momentum"]
                    m.mul_(beta).add_(grad, alpha=1.0 - dampening)

                    if nesterov:
                        u = grad.add(m, alpha=beta)
                    else:
                        u = m

                else:
                    u = grad

                p.sub_(u, alpha=lr)

        return loss


class Anchored_SGD(Optimizer):
    r"""Implements Anchored (Adaptive) Stochastic Gradient Descent (ASGD) optimizer (tentative name).

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        meta_lr (float, optional): Meta learning rate to learn the learning rate (tensor-wise) (default: 1e-3).
        min_meta_lr (float, optional): Minimum meta learning rate value (default: 1e-7).
        beta (float, optional): Momentum factor (default: 0.0).
        dampening (float, optional): Dampening for momentum (default: 0.0).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        nesterov (bool, optional): Enables Nesterov momentum (default: False).

    TODO: this is currently ONLY boilerplate; actual logic yet to be implemented!
    """
    def __init__(
            self,
            params,
            meta_lr: float = 1e-3,
            min_meta_lr: float = 1e-7,
            beta: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
            ):
        if meta_lr <= 0.0:
            raise ValueError(f"Invalid meta learning rate: {meta_lr}")
        if min_meta_lr <= 0.0:
            raise ValueError(f"Invalid minimum meta learning rate: {meta_lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid momentum value: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
        if nesterov and (beta <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            meta_lr=meta_lr,
            min_meta_lr=min_meta_lr,
            momentum=beta,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            )
        super().__init__(params, defaults)  # exposes "self.param_groups"

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            The loss returned by the closure, if any.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            meta_lr = group["meta_lr"]
            min_meta_lr = group["min_meta_lr"]
            beta = group["beta"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay:
                    grad = grad.add(p, alpha=weight_decay)

                if beta:
                    param_state = self.state[p]
                    if "momentum" not in param_state:
                        param_state["momentum"] = torch.zeros_like(grad)

                    m = param_state["momentum"]
                    m.mul_(beta).add_(grad, alpha=1.0 - dampening)

                    if nesterov:
                        u = grad.add(m, alpha=beta)
                    else:
                        u = m

                else:
                    u = grad

                p.sub_(u, alpha=meta_lr)

        return loss
