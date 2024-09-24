import torch

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        device = full_rank_grad.device
        dtype = full_rank_grad.dtype

        if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
            matrix_type = self._get_matrix_type(full_rank_grad)
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, matrix_type)
            # Move ortho_matrix to the same device and dtype as full_rank_grad
            if isinstance(self.ortho_matrix, list):
                self.ortho_matrix = [m.to(device=device, dtype=dtype) for m in self.ortho_matrix]
            else:
                self.ortho_matrix = self.ortho_matrix.to(device=device, dtype=dtype)

        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'right':
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad)
            low_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix[1].t())
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.ortho_matrix is None:
            raise RuntimeError("Ortho_matrix is not initialized. Call project() before project_back().")

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad)
            full_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix[1])
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, matrix, rank, matrix_type):
        device = matrix.device
        dtype = matrix.dtype

        # Ensure matrix is float32 for SVD computation
        if dtype != torch.float32:
            matrix = matrix.to(dtype=torch.float32)
        else:
            matrix = matrix.clone()

        with torch.no_grad():
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

            if matrix_type == 'right':
                B = Vh[:rank, :]
                return B.to(device=device, dtype=dtype)
            elif matrix_type == 'left':
                A = U[:, :rank]
                return A.to(device=device, dtype=dtype)
            elif matrix_type == 'full':
                A = U[:, :rank]
                B = Vh[:rank, :]
                return [A.to(device=device, dtype=dtype), B.to(device=device, dtype=dtype)]
            else:
                raise ValueError("matrix_type should be 'left', 'right', or 'full'")

    def _get_matrix_type(self, matrix):
        if self.proj_type == 'std':
            return 'right' if matrix.shape[0] >= matrix.shape[1] else 'left'
        elif self.proj_type == 'reverse_std':
            return 'left' if matrix.shape[0] >= matrix.shape[1] else 'right'
        elif self.proj_type in ['left', 'right', 'full']:
            return self.proj_type
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")
