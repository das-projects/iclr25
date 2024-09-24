import torch
from tensorly.decomposition import tucker
from tensorly import tenalg

# Ensure tensorly uses PyTorch as the backend
import tensorly as tl

tl.set_backend("pytorch")


# The GaLoreProjector class in Python implements a projection method using orthogonal matrix
# decomposition for low-rank approximation of gradients for general tensors.
# We use tensor decomposition using tensorly library: https://tensorly.org/stable/index.html
class GaLoreProjectorTensor:
    """
    GaLore Projector for tensors using tensor decomposition.

    Args:
        rank (int): The rank for the Tucker decomposition.
        update_proj_gap (int, optional): Iterations between updating the projection factors. Defaults to 200.
        scale (float, optional): Scaling factor for the projected gradients. Defaults to 1.0.
        proj_type (str, optional): Type of projection ('std' or 'continuous'). Defaults to 'std'.
    """

    def __init__(self, rank, update_proj_gap=200, scale=1.0, proj_type="galore"):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.core = None
        self.factors = None
        self.optimizers = None  # For continuous projection
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter, lr_ratio=1.0):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.
            lr_ratio (float): Ratio of current learning rate to initial learning rate.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        match self.proj_type:
            case "galore":
                if self.core is None or iter % self.update_proj_gap == 0:
                    self.core, self.factors = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank
                    )
                self.transformed_low_rank = self.transform(self.factors, full_rank_grad)
            case "online_galore":
                self.transformed_low_rank = self._project_online_galore(full_rank_grad, lr_ratio)
            case "online_natural_galore":
                self.transformed_low_rank = self._project_online_galore(full_rank_grad, lr_ratio)
                self.transformed_low_rank = self.natural_gradient_transform(self.transformed_low_rank)
            case _:
                raise NotImplementedError(
                    f"Projection type '{self.proj_type}' is not implemented."
                )
        return self.transformed_low_rank

    def _project_online_galore(self, full_rank_grad, lr_ratio):
        """
        Projects the full-rank gradients onto the low-rank subspace using online GaLore.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            lr_ratio (float): Ratio of current learning rate to initial learning rate.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.core is None:
            self.core, self.factors = self.get_orthogonal_matrix(
                full_rank_grad, self.rank
            )
            # Initialize optimizers for factors
            self.optimizers = []
            for factor in self.factors:
                factor.requires_grad = True
                optimizer = torch.optim.AdamW([factor], lr=1.0 / self.update_proj_gap)
                self.optimizers.append(optimizer)
        else:
            self._update_factors(full_rank_grad, lr_ratio)
        return self.transform(self.factors, full_rank_grad)

    def natural_gradient_transform(self, low_rank_grad):
        """
        Transforms the low-rank gradients using the natural gradient.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        # Compute the natural gradient
        pass

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(self.factors, self.transformed_low_rank)
        if "galore" != self.proj_type:
            # Update factors after projection back
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
        return full_rank_grad * self.scale

    def _update_factors(self, full_rank_grad, lr_ratio):
        """
        Updates the factors using gradient descent.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            lr_ratio (float): Ratio of current learning rate to initial learning rate.
        """
        # Set all factors to require gradients
        for factor in self.factors:
            factor.requires_grad = True

        # Zero gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Compute loss for updating factors
        approx_grad = self.inverse_transform(
            self.factors, self.transform(self.factors, full_rank_grad)
        )
        loss = torch.norm(full_rank_grad - approx_grad) ** 2

        # Backpropagate
        loss.backward()

        # Update learning rates based on lr_ratio
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = (1.0 / self.update_proj_gap) * lr_ratio

        # Note: Optimizer steps are taken in project_back after projection

    def get_orthogonal_matrix(self, weights, rank):
        """
        Computes the Tucker decomposition of the weights.

        Args:
            weights (torch.Tensor): The weights to decompose.
            rank (int or tuple): The desired rank for each mode.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The core tensor and factors.
        """
        # Ensure weights are float32
        original_dtype = weights.dtype
        if weights.dtype != torch.float32:
            weights = weights.float()

        # Determine the ranks for each mode
        if isinstance(rank, int):
            # For higher-order tensors, apply the same rank to all modes
            ranks = [rank] * weights.ndim
        elif isinstance(rank, (list, tuple)):
            # Ensure that the length of rank matches the number of dimensions
            if len(rank) != weights.ndim:
                raise ValueError(
                    f"Rank tuple length {len(rank)} does not match tensor dimensions {weights.ndim}."
                )
            ranks = rank
        else:
            raise ValueError("Rank must be an integer or a tuple/list of integers.")

        # Perform Tucker decomposition
        core, factors = tucker(weights, ranks=ranks)

        # Convert factors back to original dtype if necessary
        factors = [factor.to(dtype=original_dtype) for factor in factors]

        return core, factors

    def transform(self, factors, x):
        """
        Transforms the input tensor using the factors.

        Args:
            factors (List[torch.Tensor]): Factors from the Tucker decomposition.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        """
        Reconstructs the tensor from the low-rank representation.

        Args:
            factors (List[torch.Tensor]): Factors from the Tucker decomposition.
            x (torch.Tensor): The low-rank representation.

        Returns:
            torch.Tensor: The reconstructed tensor.
        """
        return tenalg.multi_mode_dot(x, factors)
