import torch
import torch.nn as nn
from tqdm import tqdm

class RobustPCA(nn.Module):
    """
    RPCA: Robust Principal Component Analysis.

    Based on Candès, Li, Ma, and Wright ("Robust Principal Component
    Analysis?").
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
        L, _ = self.decompose_ialm(x)
        return L

    def decompose_ialm(self, x):
        original_shape = x.shape

        flat_x = x.view(x.size(0), -1)
        
        x_mat = flat_x
            
        x_mat_norm = torch.norm(x_mat, 'fro')
        
        n1, n2 = torch.tensor(x_mat.shape)
        
        # Parameter defaults suggested in the paper
        if self.lambda_ is None:
            self.lambda_ = 1.0 / torch.sqrt(max(n1, n2))
        
        self.mu = 1.0 / torch.norm(x_mat, 2) # Starting small encourages exploration
        rho = 1.5 # Growth factor (standard heuristic)
        mu_bar = 1e7 # Maximum mu

        L = torch.zeros_like(x_mat)
        S = torch.zeros_like(x_mat)
        Y = torch.zeros_like(x_mat)
        
        loop = tqdm(range(self.max_iter), desc="RPCA Optimization", leave=True)
        min_error = float('inf')
        
        for k in loop:
            
            # Update L (Low Rank) using Singular Value Thresholding
            temp_L = x_mat - S + (1/self.mu) * Y
            u, s, v = torch.linalg.svd(temp_L, full_matrices=False)
            s_thresh = self.soft_threshold(s, 1/self.mu)
            L = u @ torch.diag_embed(s_thresh) @ v
            
            # Update S (Sparse) using Shrinkage 
            temp_S = x_mat - L + (1/self.mu) * Y
            S = self.soft_threshold(temp_S, self.lambda_ / self.mu)
            
            # Update Y (Lagrange Multiplier)
            residual = x_mat - L - S
            Y = Y + self.mu * residual
            self.mu = min(self.mu * rho, mu_bar)
            
            # Convergence check
            res_norm = torch.norm(residual, 'fro')
            error = res_norm / x_mat_norm
            min_error = min(min_error, error)
            
            loop.set_postfix({'Error': f'{error.item():.2e}', 'Min Error': f'{min_error.item():.2e}'})
            if error <= self.tol:
                loop.close()
                print(f"RPCA converged in {k+1} iterations with error {error.item():.2e}")
                break
        
        return L.view(original_shape), S.view(original_shape)

    def decompose_ealm(self, x):
        original_shape = x.shape
        flat_x = x.view(x.size(0), -1)
        
        x_mat = flat_x
            
        x_mat_norm = torch.norm(x_mat, 'fro')
        n1, n2 = torch.tensor(x_mat.shape)
        
        # Standard parameters
        if self.lambda_ is None:
            self.lambda_ = 1.0 / torch.sqrt(max(n1, n2))
            
        # Fixed mu for Exact ALM (no continuation scaling)
        if self.mu is None:
            self.mu = (n1 * n2) / (4.0 * torch.norm(x_mat, 1))

        L = torch.zeros_like(x_mat)
        S = torch.zeros_like(x_mat)
        Y = torch.zeros_like(x_mat)
        
        loop = tqdm(range(self.max_iter), desc="Pure Exact ALM Optimization", leave=True)
        min_error = float('inf')
        
        inner_tol = 1e-5 
        max_inner_iter = 100 

        for k in loop:
            
            L_inner = L.clone()
            S_inner = S.clone()
            
            for j in range(max_inner_iter):
                L_prev = L_inner.clone()
                S_prev = S_inner.clone()
                
                # Update L
                temp_L = x_mat - S_inner + (1/self.mu) * Y
                u, s, v = torch.linalg.svd(temp_L, full_matrices=False)
                s_thresh = self.soft_threshold(s, 1/self.mu)
                L_inner = u @ torch.diag_embed(s_thresh) @ v
                
                # Update S
                temp_S = x_mat - L_inner + (1/self.mu) * Y
                S_inner = self.soft_threshold(temp_S, self.lambda_ / self.mu)
                
                # Check inner convergence
                diff_L = torch.norm(L_inner - L_prev, 'fro') / max(1e-8, torch.norm(L_prev, 'fro'))
                diff_S = torch.norm(S_inner - S_prev, 'fro') / max(1e-8, torch.norm(S_prev, 'fro'))
                
                if diff_L < inner_tol and diff_S < inner_tol:
                    break
            
            L = L_inner
            S = S_inner

            residual = x_mat - L - S
            Y = Y + self.mu * residual
            
            # Outer Convergence check
            res_norm = torch.norm(residual, 'fro')
            error = res_norm / x_mat_norm
            min_error = min(min_error, error)
            
            loop.set_postfix({'Error': f'{error.item():.2e}', 'Min Error': f'{min_error.item():.2e}'})
            
            if error <= self.tol:
                loop.close()
                print(f"Pure Exact ALM converged in {k+1} outer iterations with error {error.item():.2e}")
                break
        
        
        return L.view(original_shape), S.view(original_shape)