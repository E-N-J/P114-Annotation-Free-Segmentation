import torch
import numpy as np
from tqdm import tqdm

class RobustPCA:
    """
    Implements Robust PCA using the Augmented Lagrange Multiplier method
    described in Candès et al., Section 5[cite: 771].
    """
    def __init__(self, lambda_=None, mu=None, tol=1.06e-6, max_iter=100, device=torch.device("cpu")):
        self.lambda_ = lambda_
        self.mu = mu
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        
    def soft_threshold(self, z, tau):
        return torch.sign(z) * torch.maximum(torch.abs(z) - tau, torch.tensor(0.0).to(z.device))

    def decompose(self, x, fast=False):
        """
        Input x: (Batch, Channels, Height, Width) or (Batch, Flattened)
        """
        x = x.to(self.device)
        original_shape = x.shape
        # RPCA usually works on a 2D matrix. 
        # Option A: Stack batch as columns (standard video/stack approach)
        # Option B: Flatten each image (standard single image decomposition)
        # Here we flatten the spatial dims: (Batch, Features)
        x_mat = x.view(x.size(0), -1) 
        
        n1, n2 = x_mat.shape
        
        # Parameter defaults suggested in the paper [cite: 780]
        if self.lambda_ is None:
            self.lambda_ = 1.0 / np.sqrt(max(n1, n2))
        
        # Suggested mu in paper [cite: 779]
        if self.mu is None:
            self.mu = (n1 * n2) / (4.0 * torch.norm(x_mat, 1))
            
        if fast:
            self.mu = 1.0 / torch.norm(x_mat, 2) # Starting small encourages exploration
            rho = 1.5 # Growth factor (standard heuristic)
            mu_bar = 1e7 # Maximum mu

        L = torch.zeros_like(x_mat).to(self.device)
        S = torch.zeros_like(x_mat).to(self.device)
        Y = torch.zeros_like(x_mat).to(self.device)
        
        loop = tqdm(range(self.max_iter), desc="RPCA Optimization", leave=True)
        min_error = float('inf')
        
        for k in loop:
            
            # 1. Update L (Low Rank) using Singular Value Thresholding [cite: 777]
            temp_L = x_mat - S + (1/self.mu) * Y
            u, s, v = torch.linalg.svd(temp_L, full_matrices=False)
            s_thresh = self.soft_threshold(s, 1/self.mu)
            L = u @ torch.diag_embed(s_thresh) @ v
            
            # 2. Update S (Sparse) using Shrinkage 
            temp_S = x_mat - L + (1/self.mu) * Y
            S = self.soft_threshold(temp_S, self.lambda_ / self.mu)
            
            # 3. Update Y (Lagrange Multiplier) [cite: 771]
            residual = x_mat - L - S
            Y = Y + self.mu * residual
            
            if fast:
                self.mu = min(self.mu * rho, mu_bar)
            
            # 4. Convergence check [cite: 779]
            error = torch.norm(residual, 'fro') / torch.norm(x_mat, 'fro')
            min_error = min(min_error, error)
            
            loop.set_postfix({'Error': f'{error.item():.2e}', 'Min Error': f'{min_error.item():.2e}'})
            if error < self.tol:
                loop.close()
                print(f"RPCA converged in {k+1} iterations with error {error.item():.2e}")
                break
        
        return L.view(original_shape), S.view(original_shape)

