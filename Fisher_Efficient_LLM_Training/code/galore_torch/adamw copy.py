# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import List, Optional, Tuple, Union, Callable, Iterable

import math
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable, _get_value
from torch.optim.optimizer import _foreach_doc, _fused_doc, _capturable_doc, _maximize_doc, _differentiable_doc
from torch.optim.optimizer import _get_capturable_supported_devices, _default_to_fused_or_foreach, _stack_if_compiling

# Import GaLoreProjector and GaLoreProjectorTensor
from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor

__all__ = ["GaLore", "galore"]


class GaLore(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        *,
        rank: Optional[int] = None,
        update_proj_gap: int = 1,
        scale: float = 1.0,
        proj_type: str = "gaussian",
        dim: int = 2,
        correct_bias: bool = True,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
            dim=dim,
            correct_bias=correct_bias,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if len(p_state) != 0 and not torch.is_tensor(p_state.get("step", None)):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=torch.float, device=p.device)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        projectors,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("GaLore does not support sparse gradients")
            grads.append(grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                if group["fused"]:
                    if p.device.type != "cuda":
                        raise RuntimeError("Fused optimizer requires CUDA devices")
                state["step"] = torch.zeros((), dtype=torch.float, device=p.device)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Initialize projector
                if group["rank"] is not None:
                    if group["dim"] <= 2:
                        state["projector"] = GaLoreProjector(
                            group["rank"],
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"],
                            proj_type=group["proj_type"],
                        )
                    else:
                        state["projector"] = GaLoreProjectorTensor(
                            group["rank"],
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"],
                            proj_type=group["proj_type"],
                        )
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
            projector = state.get("projector", None)
            projectors.append(projector)

    @_use_grad_for_differentiable
    def step(self, closure: Callable = None):
        """Performs a single optimization step."""
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            projectors: List[Optional[Callable]] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                projectors,
            )

            galore(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                projectors,
                beta1=group["betas"][0],
                beta2=group["betas"][1],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                correct_bias=group["correct_bias"],
            )

        return loss


def _single_tensor_galore(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    projectors: List[Optional[Callable]],
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    correct_bias: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        if maximize:
            grad = -grad
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        projector = projectors[i]

        if projector is not None:
            grad = projector.project(grad, _get_value(step_t))

        # Update step
        step_t += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if correct_bias:
            bias_correction1 = 1 - beta1 ** step_t
            bias_correction2 = 1 - beta2 ** step_t
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        else:
            step_size = lr

        denom = exp_avg_sq.sqrt().add_(eps)

        norm_grad = exp_avg / denom

        if projector is not None:
            norm_grad = projector.project_back(norm_grad)

        param.add_(norm_grad, alpha=-step_size)

        if weight_decay != 0:
            param.add_(param, alpha=-lr * weight_decay)


def _multi_tensor_galore(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    projectors: List[Optional[Callable]],
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    correct_bias: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # Apply GaLore projection to gradients
    for i in range(len(grads)):
        if projectors[i] is not None:
            grads[i] = projectors[i].project(grads[i], _get_value(state_steps[i]))
        if maximize:
            grads[i] = -grads[i]

    # Update steps
    torch._foreach_add_(state_steps, 1)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    # Compute bias corrections
    if correct_bias:
        bias_correction1 = [1 - beta1 ** _get_value(step) for step in state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in state_steps]
        step_size = [
            lr * math.sqrt(bc2) / bc1 for bc1, bc2 in zip(bias_correction1, bias_correction2)
        ]
    else:
        step_size = [lr] * len(state_steps)

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_add_(denom, eps)

    norm_grads = []
    for exp_avg, denom_i in zip(exp_avgs, denom):
        norm_grad = exp_avg / denom_i
        norm_grads.append(norm_grad)

    # Apply GaLore projection back
    for i in range(len(norm_grads)):
        if projectors[i] is not None:
            norm_grads[i] = projectors[i].project_back(norm_grads[i])

    torch._foreach_add_(params, norm_grads, alpha=-1)

    if weight_decay != 0:
        torch._foreach_add_(params, params, alpha=-lr * weight_decay)


def _fused_galore(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    projectors: List[Optional[Callable]],
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    correct_bias: bool,
):
    raise NotImplementedError(
        "Fused implementation is not supported for GaLore optimizer due to custom projection operations."
    )


def galore(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    projectors: List[Optional[Callable]],
    # Keyword-only arguments
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    correct_bias: bool,
):
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not enable foreach if custom projections are used
        foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if fused:
        func = _fused_galore
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_galore
    else:
        func = _single_tensor_galore

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        projectors,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        correct_bias=correct_bias,
    )
