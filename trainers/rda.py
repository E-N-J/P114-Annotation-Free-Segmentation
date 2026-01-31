import torch
import torch.nn as nn
import torch.optim as optim
from trainers.base import BaseTrainer
from .utils import shrinkage_l1, shrinkage_l21

class RDATrainer(BaseTrainer):
    """
    Trainer for Robust Deep Autoencoders using ADMM-like alternating minimization.
    """
    def __init__(self, model, loader):
        super().__init__(model, loader)
        self.criterion = nn.MSELoss()
        
        # Data-dependent initializations
        num_samples = len(self.loader.dataset)
        sample_img, _, _ = self.loader.dataset[0]
        c, h, w = sample_img.shape
        self.pixels_per_img = c * h * w
        
        # Initialize S and LS to zero
        self.S_memory = torch.zeros(num_samples, c, h, w).to(self.device)
        self.LS_memory = torch.zeros(num_samples, c, h, w).to(self.device)
        
        # Pre-calculate Norm of X for convergence checks
        X_norm_sq = 0.0
        for x, _, indices in self.loader:
            x = x.to(self.device)
            indices = indices.to(self.device)
            self.LS_memory[indices] = x 
            X_norm_sq += torch.sum(x ** 2).item() 
            
        self.X_frobenius = X_norm_sq ** 0.5 
        self.total_pixels = num_samples * self.pixels_per_img

    def fit(self, lr=1e-3, lambda_=1.0, outer_epochs=20, inner_epochs=50, tol=1e-7, norm_type='l1'):
        """
        Implementation of ADMM for an RDA.
        Decomposes input data X into Low-Rank L and Sparse S components.
        """
        print(f"\nTraining with ADMM Algorithm ({norm_type} norm)...")
        
        optimiser = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        outer_pbar = self.tqdm(range(outer_epochs), desc="ADMM Steps", leave=True)
        
        for out_step in outer_pbar:
            
            # --- Phase 1: Train Autoencoder (Update L) ---
            inner_pbar = self.tqdm(range(inner_epochs), desc=f"Step {out_step+1}: Training AE", leave=False)
            for in_epoch in inner_pbar:
                total_ae_loss = 0
                for x, _, indices in self.loader:
                    x = x.to(self.device)
                    indices = indices.to(self.device)
                    
                    # [Step 1]: Remove S from X
                    with torch.no_grad():
                        S_batch = self.S_memory[indices]
                        L_D = x - S_batch
                    
                    # 2. Train Network to reconstruct X - S
                    optimiser.zero_grad()
                    L_D_output = self.model(L_D)
                    
                    ae_loss = self.criterion(L_D_output, L_D)
                    ae_loss.backward()
                    optimiser.step()
                    
                    total_ae_loss += ae_loss.item()
                
                # Report Avg AE Loss
                avg_ae_loss = total_ae_loss / len(self.loader)
                self.histories['ae_loss'] = self.histories.get('ae_loss',[]) + [avg_ae_loss]
                self.global_step += 1
                
                inner_pbar.set_postfix({'L_Rec': f"{avg_ae_loss:.2e}" })
                
                if self.is_notebook:
                    self.plot_metrics(log_scale=True)
            
            # --- Phase 2: Update S Matrix ---
            constraint_error = 0.0
            convergence_gap = 0.0
            total_sparsity_sum = 0.0 
            total_L_rec_loss = 0.0
            
            with torch.no_grad():
                for x, _, indices in self.loader:
                    x = x.to(self.device)
                    indices = indices.to(self.device)
                    
                    S_old = self.S_memory[indices]
                    L_D_input = x - S_old
                    
                    # [Step 3]: L_D = D(E(L_D))
                    L_D_output = self.model(L_D_input)
                    
                    # [Step 4]: S_prox_input = X - L_D
                    S_prox_input = x - L_D_output
                    
                    # [Step 5]: Optimize S
                    if norm_type == 'l1':
                        S_new = shrinkage_l1(S_prox_input, lambda_)
                        sparsity_sum = torch.sum(torch.abs(S_new))
                    elif norm_type == 'l21':
                        res_flat = S_prox_input.view(S_prox_input.size(0), -1)
                        S_flat = shrinkage_l21(res_flat, lambda_)
                        S_new = S_flat.view_as(S_prox_input)
                        col_norms = torch.norm(S_new.view(x.shape[0], -1), p=2, dim=0)
                        sparsity_sum = torch.sum(col_norms)
                    
                    self.S_memory[indices] = S_new
                    
                    # [Step 7 Prep]: Constraint c1 (Reconstruction Error)
                    x_rec_loss = torch.sum((S_prox_input - S_new) ** 2)
                    constraint_error += x_rec_loss.item()
                    
                    # [Step 7 Prep]: Convergence c2 (Change in L+S)
                    LS_prev = self.LS_memory[indices]
                    LS_new = L_D_output + S_new
                    self.LS_memory[indices] = LS_new
                    convergence_gap += torch.sum((LS_prev - LS_new) ** 2).item()
                    
                    # Accumulate for Reporting
                    total_sparsity_sum += sparsity_sum.item()
                    total_L_rec_loss += torch.sum((L_D_input - L_D_output) ** 2).item()
            
            # [Step 6]: Convergence Checks
            c1 = (constraint_error ** 0.5) / (self.X_frobenius + 1e-9)
            c2 = (convergence_gap ** 0.5) / (self.X_frobenius + 1e-9)
            
            # Compute Averages and Objective for Reporting                    
            sparsity_term = total_sparsity_sum / self.total_pixels
            avg_L_rec_loss = total_L_rec_loss / self.total_pixels
            objective = avg_L_rec_loss + lambda_ * sparsity_term
            
            # Report Metrics
            self.histories["c1"] = self.histories.get("c1", []) + inner_epochs * [c1]
            self.histories["c2"] = self.histories.get("c2", []) + inner_epochs * [c2]
            self.histories['sparsity_term'] = self.histories.get('sparsity_term', []) + inner_epochs * [sparsity_term]
            self.histories['objective'] = self.histories.get('objective', []) + inner_epochs * [objective]
            sparsity_pct = (self.S_memory != 0).float().mean()
            
            outer_pbar.set_postfix({
                'Obj': f"{objective:.1e}",
                'L_Rec': f"{avg_L_rec_loss:.1e}",
                'Sparse': f"{sparsity_term:.1e}",
                'S%': f"{sparsity_pct.item()*100:.1f}%"
            })
            
            if self.is_notebook:
                self.plot_metrics(log_scale=True)
                
            if c1 < tol or c2 < tol:
                print(f"\nConverged! c1={c1:.2e}, c2={c2:.2e}")
                break
            
        print(f"Final Reconstruction Loss: {avg_ae_loss:.4e}")      
        print("ADMM Training Complete.")
