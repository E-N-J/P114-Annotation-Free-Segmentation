import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class RobustPCA(nn.Module):
    """
    Implements Robust PCA using the Augmented Lagrange Multiplier method
    described in Candes et al., Section 5.
    """
    def __init__(self, lambda_=None, mu=None, tol=1.06e-6, max_iter=100):
        super().__init__()
        self.lambda_ = lambda_
        self.mu = mu
        self.tol = tol
        self.max_iter = max_iter
        self.register_buffer('dummy_buffer', torch.empty(0))
        
    def soft_threshold(self, z, tau):
        return torch.sign(z) * torch.maximum(torch.abs(z) - tau, torch.tensor(0.0).to(z.device))

    def forward(self, x):
        """
        Adapter method to make RPCA compatible with Neural Network evaluation loops.
        Returns L (Low Rank / Background).
        """
        L, _ = self.decompose(x, fast=True, cols=False)
        return L

    def decompose(self, x, fast=False, cols=False):
        """
        Input x: (Batch, Channels, Height, Width) or (Batch, Flattened)
        """
        original_shape = x.shape

        flat_x = x.view(x.size(0), -1)
       
        if cols:
            x_mat = flat_x.t() 
        else:
            x_mat = flat_x
            
        x_mat_norm = torch.norm(x_mat, 'fro')
        
        n1, n2 = x_mat.shape
        
        # Parameter defaults suggested in the paper
        if self.lambda_ is None:
            self.lambda_ = 1.0 / np.sqrt(max(n1, n2))
        
        # Suggested mu in paper
        if self.mu is None:
            self.mu = (n1 * n2) / (4.0 * torch.norm(x_mat, 1))
            
        if fast:
            self.mu = 1.0 / torch.norm(x_mat, 2) # Starting small encourages exploration
            rho = 1.5 # Growth factor (standard heuristic)
            mu_bar = 1e7 # Maximum mu

        L = torch.zeros_like(x_mat)
        S = torch.zeros_like(x_mat)
        Y = torch.zeros_like(x_mat)
        
        loop = tqdm(range(self.max_iter), desc="RPCA Optimization", leave=True)
        min_error = float('inf')
        
        for k in loop:
            
            # [Step 1]: Update L (Low Rank) using Singular Value Thresholding
            temp_L = x_mat - S + (1/self.mu) * Y
            u, s, v = torch.linalg.svd(temp_L, full_matrices=False)
            s_thresh = self.soft_threshold(s, 1/self.mu)
            L = u @ torch.diag_embed(s_thresh) @ v
            
            # [Step 2]: Update S (Sparse) using Shrinkage 
            temp_S = x_mat - L + (1/self.mu) * Y
            S = self.soft_threshold(temp_S, self.lambda_ / self.mu)
            
            # [Step 3]: Update Y (Lagrange Multiplier)
            residual = x_mat - L - S
            Y = Y + self.mu * residual
            
            if fast:
                self.mu = min(self.mu * rho, mu_bar)
            
            # [Step 4]: Convergence check
            res_norm = torch.norm(residual, 'fro')
            error = res_norm / x_mat_norm
            min_error = min(min_error, error)
            
            loop.set_postfix({'Error': f'{error.item():.2e}', 'Min Error': f'{min_error.item():.2e}'})
            if error <= self.tol:
                loop.close()
                print(f"RPCA converged in {k+1} iterations with error {error.item():.2e}")
                break
            
        if cols:
            L = L.t()
            S = S.t()
        
        return L.view(original_shape), S.view(original_shape)
