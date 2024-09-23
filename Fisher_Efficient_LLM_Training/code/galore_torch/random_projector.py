import torch

torch_compile_options = {
    "epilogue_fusion": False,
    "max_autotune": False,
    "shape_padding": True,
    "trace.enabled": False,  # Output Triton kernel outputs!
    "triton.cudagraphs": True,
}


class RandomProjector:
    def __init__(
        self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type="std"
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                )
                low_rank_grad = torch.matmul(
                    full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type)
                )
            else:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                )
                low_rank_grad = torch.matmul(
                    self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad
                )
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                )
                low_rank_grad = torch.matmul(
                    self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad
                )
            else:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                )
                low_rank_grad = torch.matmul(
                    full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type)
                )
        elif self.proj_type == "right":
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad, self.rank, type="right"
            )
            low_rank_grad = torch.matmul(
                full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type)
            )
        elif self.proj_type == "left":
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad, self.rank, type="left"
            )
            low_rank_grad = torch.matmul(
                self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad
            )
        elif self.proj_type == "full":
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad, self.rank, type="full"
            )
            low_rank_grad = torch.matmul(
                self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad
            ) @ self.ortho_matrix[1].t().to(full_rank_grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
                )
            else:
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
                )
        elif self.proj_type == "reverse_std":
            if (
                low_rank_grad.shape[0] <= low_rank_grad.shape[1]
            ):  # note this is different from std
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
                )
            else:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
                )
        elif self.proj_type == "right":
            full_rank_grad = torch.matmul(
                low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
            )
        elif self.proj_type == "left":
            full_rank_grad = torch.matmul(
                self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
            )
        elif self.proj_type == "full":
            full_rank_grad = torch.matmul(
                self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad
            ) @ self.ortho_matrix[1].to(low_rank_grad.device.type)

        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        # make the smaller matrix always to be orthogonal matrix
        if type == "right":
            B = torch.randn(matrix.shape[1], self.rank) / self.rank
            B = gram_schmidt(B)
            B = B.t()
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == "left":
            A = torch.randn(matrix.shape[0], self.rank) / self.rank
            A = gram_schmidt(A)
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == "full":
            A = torch.randn(matrix.shape[0], self.rank) / self.rank
            B = torch.randn(matrix.shape[1], self.rank) / self.rank
            A = gram_schmidt(A)
            B = gram_schmidt(B)
            B = B.t()
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError("type should be left, right or full")


@torch.compile(fullgraph=True, options=torch_compile_options)
def gram_schmidt(A):
    """Perform Gram-Schmidt orthogonalization on matrix A.
    Args:
        A (torch.Tensor): Input matrix of size (n, n) with linearly independent columns.
    Returns:
        Q (torch.Tensor): Orthogonal matrix of the same size as A.
    """
    _, n = A.shape
    Q = torch.zeros_like(A)

    for i in range(n):
        # Take the current column of A
        v_i = A[:, i]

        # Subtract the projection of v_i onto the previous q_j's
        for j in range(i):
            q_j = Q[:, j]
            v_i = v_i - torch.dot(q_j, v_i) * q_j

        # Normalize v_i to get the orthonormal vector q_i
        q_i = v_i / torch.norm(v_i)
        Q[:, i] = q_i

    return Q
