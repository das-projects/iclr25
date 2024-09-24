import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple, Union, Iterable, Callable


class GaLoreProjector:
    def __init__(self, rank, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.ortho_matrix = None
        self.ortho_matrix_optim = None

    def project(self, full_rank_grad, step, lr_ratio=1.0):
        if self.proj_type == 'std':
            low_rank_grad = self._std_projection(full_rank_grad, step)
        elif 'continuous' in self.proj_type:
            low_rank_grad = self._continuous_projection(full_rank_grad, step, lr_ratio)
        else:
            raise NotImplementedError(f"Projection type '{self.proj_type}' is not implemented.")
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            full_rank_grad = self._std_project_back(low_rank_grad)
        elif 'continuous' in self.proj_type:
            full_rank_grad = self._continuous_project_back(low_rank_grad)
        else:
            raise NotImplementedError(f"Projection type '{self.proj_type}' is not implemented.")
        return full_rank_grad * self.scale

    def _std_projection(self, full_rank_grad, step):
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode='right'
                )
            return torch.matmul(full_rank_grad, self.ortho_matrix.t())
        else:
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode='left'
                )
            return torch.matmul(self.ortho_matrix.t(), full_rank_grad)

    def _std_project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            return torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            return torch.matmul(self.ortho_matrix, low_rank_grad)

    def _continuous_projection(self, full_rank_grad, step, lr_ratio):
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode='right'
                )
                self.ortho_matrix.requires_grad = True
                self.ortho_matrix_optim = torch.optim.AdamW(
                    [self.ortho_matrix], lr=1 / self.update_proj_gap
                )
            else:
                if step % self.update_proj_gap == 0:
                    self._update_ortho_matrix(full_rank_grad, lr_ratio)
            return torch.matmul(full_rank_grad, self.ortho_matrix.t())
        else:
            if self.ortho_matrix is None:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, mode='left'
                )
                self.ortho_matrix.requires_grad = True
                self.ortho_matrix_optim = torch.optim.AdamW(
                    [self.ortho_matrix], lr=1 / self.update_proj_gap
                )
            else:
                if step % self.update_proj_gap == 0:
                    self._update_ortho_matrix(full_rank_grad, lr_ratio)
            return torch.matmul(self.ortho_matrix.t(), full_rank_grad)

    def _continuous_project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        # Update the ortho_matrix optimizer
        self.ortho_matrix_optim.step()
        self.ortho_matrix.grad = None
        return full_rank_grad

    def _update_ortho_matrix(self, full_rank_grad, lr_ratio):
        self.ortho_matrix.grad = None
        projection = self.ortho_matrix @ self.ortho_matrix.t()
        normalized_grad = full_rank_grad / full_rank_grad.norm()
        loss = (normalized_grad - projection @ normalized_grad).norm() ** 2
        loss.backward()
        # Update optimizer learning rate based on lr_ratio
        for group in self.ortho_matrix_optim.param_groups:
            group['lr'] = (1 / self.update_proj_gap) * lr_ratio
        self.ortho_matrix_optim.step()
        self.ortho_matrix.grad = None

    def get_orthogonal_matrix(self, weights, rank, mode):
        U, s, Vh = torch.linalg.svd(weights, full_matrices=False)
        if mode == 'right':
            return Vh[:rank, :].detach()
        elif mode == 'left':
            return U[:, :rank].detach()
        else:
            raise ValueError("Mode should be 'left' or 'right'.")


class AdamW(Optimizer):
    def __init__(
        self,
        params: Union[Iterable[nn.parameter.Parameter], Iterable[dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        rank: Optional[int] = None,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = 'std',
    ):
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
            amsgrad=amsgrad,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
        )
        super().__init__(params, defaults)
        self.init_lr = lr

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            for p in group["params"]:
                state = self.state[p]
                if len(state) != 0 and not torch.is_tensor(state["step"]):
                    state["step"] = torch.tensor(float(state["step"]))

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            projectors: List[Optional[GaLoreProjector]] = []

            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # Initialize state step
                if "step" not in state:
                    state["step"] = torch.zeros(
                        (), dtype=torch.float32, device=p.device
                    ) if group["capturable"] else torch.tensor(0.0)

                # GaLore Projection
                if group["rank"] is not None:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            rank=group["rank"],
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"],
                            proj_type=group["proj_type"],
                        )
                    grad = state["projector"].project(
                        grad,
                        int(state["step"].item()),
                        lr_ratio=group["lr"] / self.init_lr,
                    )
                else:
                    state["projector"] = None

                params_with_grad.append(p)
                grads.append(grad)
                projectors.append(state["projector"])

                # State initialization
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                state_steps.append(state["step"])

            # Perform optimization step
            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                projectors,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
            )

        return loss


def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    projectors: List[Optional[GaLoreProjector]],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    """Functional API that performs AdamW algorithm computation with projection.

    Args:
        params (List[Tensor]): List of parameters.
        grads (List[Tensor]): List of gradients.
        exp_avgs (List[Tensor]): Exponential moving averages of gradients.
        exp_avg_sqs (List[Tensor]): Exponential moving averages of squared gradients.
        max_exp_avg_sqs (List[Tensor]): Max exponential moving averages of squared gradients.
        state_steps (List[Tensor]): List of step counts.
        projectors (List[Optional[GaLoreProjector]]): List of projectors per parameter.
        amsgrad (bool): Whether to use AMSGrad variant.
        beta1 (float): Coefficient for computing running averages of gradient.
        beta2 (float): Coefficient for computing running averages of squared gradient.
        lr (float): Learning rate.
        weight_decay (float): Weight decay coefficient.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the params based on the objective, instead of minimizing.
        capturable (bool): Whether to use capturable version.
        differentiable (bool): Whether to use differentiable version.
    """
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Update step
        step_t += 1

        # Perform step weight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Compute denominator
        denom = exp_avg_sq.sqrt().add_(eps)

        # Compute step size
        if capturable or differentiable:
            raise NotImplementedError(
                "Capturable or differentiable versions are not implemented."
            )
        else:
            step = step_t.item()
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2 ** 0.5
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        # Compute normalized gradient
        norm_grad = exp_avg / denom

        # GaLore Projection Back
        if projectors[i] is not None:
            norm_grad = projectors[i].project_back(norm_grad)

        # Update parameter
        param.add_(norm_grad, alpha=-step_size)
