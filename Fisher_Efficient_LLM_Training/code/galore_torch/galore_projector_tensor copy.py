import torch
from tensorly.decomposition import tucker
from tensorly import tenalg

class GaLoreProjectorTensor:
    """
    A projector class for the GaLore algorithm, handling tensors with dimensions greater than 2 using Tucker decomposition.

    Args:
        rank (int or tuple): The rank for each mode in the Tucker decomposition.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        update_proj_gap (int, optional): Number of iterations between updating the orthogonal matrices. Defaults to 200.
        scale (float, optional): Scaling factor for the projected gradients. Defaults to 1.0.
    """

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.tucker_tensor = None
        self.iter = -1  # Initialize to -1 so that first call updates the projector

    def project(self, full_rank_grad, iter):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        device = full_rank_grad.device
        dtype = full_rank_grad.dtype

        if self.tucker_tensor is None or iter - self.iter >= self.update_proj_gap:
            self.iter = iter
            self.tucker_tensor = self.get_orthogonal_matrix(full_rank_grad, self.rank)

        low_rank_grad = self.transform(self.tucker_tensor, full_rank_grad)
        return low_rank_grad

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        if self.tucker_tensor is None:
            raise RuntimeError("Tucker tensor is not initialized. Call project() before project_back().")

        full_rank_grad = self.inverse_transform(self.tucker_tensor, low_rank_grad)
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, tensor, rank):
        """
        Computes the Tucker decomposition of the tensor.

        Args:
            tensor (torch.Tensor): The tensor to decompose.
            rank (int or tuple): The desired rank for each mode.

        Returns:
            tuple: A tuple containing the core tensor and factors (core, [U1, U2, ..., Un]).
        """
        device = tensor.device
        dtype = tensor.dtype

        # Ensure tensor is in float32 for decomposition
        tensor = tensor.to(dtype=torch.float32)

        with torch.no_grad():
            # Perform Tucker decomposition
            core, factors = tucker(tensor, rank=rank)

            # Move core and factors back to the original device and dtype
            core = core.to(device=device, dtype=dtype)
            factors = [factor.to(device=device, dtype=dtype) for factor in factors]

        return (core, factors)

    def transform(self, tucker_tensor, x):
        """
        Transforms the input tensor using the factors from the Tucker decomposition.

        Args:
            tucker_tensor (tuple): The core tensor and factors from the Tucker decomposition.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        core, factors = tucker_tensor

        # Move x to the same device and dtype as core if necessary
        x = x.to(device=core.device, dtype=core.dtype)

        # Project x onto the low-rank subspace
        low_rank = tenalg.multi_mode_dot(x, factors, transpose=True)
        return low_rank

    def inverse_transform(self, tucker_tensor, x):
        """
        Reconstructs the full-rank tensor from the low-rank representation.

        Args:
            tucker_tensor (tuple): The core tensor and factors from the Tucker decomposition.
            x (torch.Tensor): The low-rank tensor.

        Returns:
            torch.Tensor: The reconstructed full-rank tensor.
        """
        core, factors = tucker_tensor

        # Move x to the same device and dtype as core if necessary
        x = x.to(device=core.device, dtype=core.dtype)

        # Reconstruct the full-rank tensor
        full_rank = tenalg.multi_mode_dot(x, factors)
        return full_rank
