import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from helpers.helper import robust_loss, shrinkage_l1, shrinkage_l21, GaussianNoise

class RobustDeepAutoencoder(nn.Module):
    """
    Robust Deep Autoencoder for decomposing images into low-rank and sparse components.
    Inspired by Zhou & Paffenroth, "Anomaly Detection with Robust Deep Autoencoders.
    """
    def __init__(self, latent_dim=3, dropout=0.1, std=0.1):
        super().__init__()
        
        # self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()
        
        self.conv_encoder = nn.Sequential(
            # 168 -> 84
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 84 -> 42 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Refinement Layer (Keep size 42x42)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.flatten_dim = 64 * 42 * 42 
        
        self.linear_encoder = nn.Sequential(
            # nn.utils.spectral_norm(nn.Linear(self.flatten_dim, 1024)),
            nn.Linear(self.flatten_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout),
            GaussianNoise(std=std),
            nn.Linear(1024, latent_dim),
            # nn.utils.spectral_norm(nn.Linear(1024, latent_dim)),
        )
        
        self.linear_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.flatten_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_decoder = nn.Sequential(
            # Input: 42x42
            # Upsample 1: 42 -> 84
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.PixelShuffle(2), # Output: 128/4 = 32 channels, size 84x84
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample 2: 84 -> 168
            nn.Conv2d(32, 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), # Output: 4/4 = 1 channel, size 168x168
            
            nn.Sigmoid()
        )
        
        self.histories = {}  

    def forward(self, x):
        x = self.conv_encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        x = self.linear_encoder(x)
        x = self.linear_decoder(x)
        
        x = x.view(-1, 64, 42, 42)
        L_pred = self.conv_decoder(x)
        
        return L_pred
    
    def simple_fit(self, loader, epochs=10, lr=1e-3, patience=5, tol=5, lambda_=0.1):
        """
        Autoencoder training with early stopping based on a simple robust loss (not specified in paper).
        """
        print(f"\nTraining Robust Autoencoder (Patience={patience})...")
        # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        device = next(self.parameters()).device
        
        self.histories = {}
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        for epoch in range(epochs): 
            total_loss = 0
            total_l2 = 0
            total_l1 = 0
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}", leave=True)
            
            for _, (x, _, _) in pbar:
                x = x.to(device)
                optimizer.zero_grad()
                
                L_pred = self(x)
                
                rob_loss, l2_loss, l1_loss = robust_loss(x, L_pred, lambda_=lambda_)
                
                rob_loss.backward()
                optimizer.step()
                
                total_loss += rob_loss.item()
                total_l2 += l2_loss.item()
                total_l1 += l1_loss.item()
                
                best_str = f"{best_loss:.4f}" if best_loss != float('inf') else "inf"
                pbar.set_postfix({
                    'total_loss': f'{rob_loss.item():.4f}',
                    'l2_loss': f'{l2_loss.item():.4f}',
                    'l1_loss': f'{l1_loss.item():.4f}',
                    'best': best_str,
                    'pat': f'{patience_counter}/{patience}'
                })
                
            avg_loss = total_loss / len(loader)
            avg_l2 = total_l2 / len(loader)
            avg_l1 = total_l1 / len(loader)
            
            if 'total_loss' not in self.histories:
                self.histories['total_loss'] = []
                self.histories['l2_loss'] = []
                self.histories['l1_loss'] = []
            
            self.histories['total_loss'].append(avg_loss)
            self.histories['l2_loss'].append(avg_l2)
            self.histories['l1_loss'].append(avg_l1)
            
            if avg_loss < best_loss - tol:
                best_loss = avg_loss
                best_avgs = (avg_loss, avg_l2, avg_l1)
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:

                    print("\nEarly stopping triggered!")
                    print(f"Restoring best model from Epoch {epoch+1 - patience} (Losses Total, (Total, L2, L1)): {best_avgs}")
                    self.load_state_dict(best_model_weights)
                    break
            
            
        print(f"Training Complete. Final Avg Losses: {best_avgs}")
            
    def plot_training_curve(self, log_scale=False):
        """
        Visualizes the training loss over epochs.
        
        Args:
            log_scale: If True, plots two graphs - one linear and one with logarithmic scale.
        """
        if not self.histories:
            print("No training history found. Run a fit first.")
            return

        num_plots = 2 if log_scale else 1
        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 5))
        
        if num_plots == 1:
            axes = [axes]
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        
        for plot_idx, ax in enumerate(axes):
            for idx, (metric_name, metric_values) in enumerate(self.histories.items()):
                if metric_values:
                    color = colors[idx % len(colors)]
                    label = metric_name.replace('_', ' ').title()
                    linestyle = '-' if metric_name == 'objective' else '--'
                    ax.plot(metric_values, linestyle=linestyle, color=color, label=label)
            
            ax.set_title('Robust Autoencoder Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            if plot_idx == 1:
                ax.set_yscale('log')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('rae_training_curve.png', dpi=300)
        plt.show()
    
    def fit_admm(self, loader, outer_epochs=20, inner_epochs=50, 
                       lr=1e-3, lambda_=1.0, tol=1e-7, norm_type='l1'):
        """
        Implementation of ADMM for an RDA as described in "Anomaly Detection with Robust Deep Autoencoders" by Zhou & Paffenroth, Section 4.1.
        Decomposes input data X into Low-Rank L and Sparse S components using an autoencoder for L and shrinkage for S.
        """
        print(f"\nTraining with Paper's Algorithm 4.1 ({norm_type} norm)...")
        
        # indexed_data = IndexedDataset(dataset)
        # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        device = next(self.parameters()).device
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        num_samples = len(loader.dataset)
        sample_img, _, _ = loader.dataset[0]
        c, h, w = sample_img.shape
        pixels_per_img = c * h * w
        
        # Initialize S and LS to zero
        self.S_memory = torch.zeros(num_samples, c, h, w).to(device)
        self.LS_memory = torch.zeros(num_samples, c, h, w).to(device)
        
        # Pre-calculate Norm of X for convergence checks
        X_norm_sq = 0.0
        
        for x, _, indices in loader:
            x = x.to(device)
            indices = indices.to(device)
            self.LS_memory[indices] = x 
            X_norm_sq += torch.sum(x ** 2).item() 
            
        X_frobenius = X_norm_sq ** 0.5 
        
        
        total_pixels = num_samples * pixels_per_img

        outer_pbar = tqdm(range(outer_epochs), desc="ADMM Steps", leave=True)
        
        for out_step in outer_pbar:
            
            # [Step 1 & 2]: Train AE on "Clean" Data
            inner_pbar = tqdm(range(inner_epochs), desc=f"Step {out_step+1}: Training AE", leave=False)
            for in_epoch in inner_pbar:
                total_ae_loss = 0
                for x, _, indices in loader:
                    x = x.to(device)
                    indices = indices.to(device)
                    
                    # [Step 1]: Remove S from X
                    with torch.no_grad():
                        S_batch = self.S_memory[indices]
                        L_D = x - S_batch
                    
                    # [Step 2]: Update AE weights
                    optimizer.zero_grad()
                    L_D_output = self(L_D)
                    
                    ae_loss = self.loss(L_D_output, L_D)

                    ae_loss.backward()
                    optimizer.step()
                    
                    total_ae_loss += ae_loss.item()
                
                # Report Avg AE Loss
                avg_ae_loss = total_ae_loss / len(loader)
                self.histories['ae_loss'] = self.histories.get('ae_loss',[]) + [avg_ae_loss]
                
                inner_pbar.set_postfix({'L_Rec': f"{avg_ae_loss:.2e}" })
                
            # [Step 3, 4, 5]: Update S Matrix
            constraint_error = 0.0
            convergence_gap = 0.0
            total_sparsity_sum = 0.0 # Sum Abs (L1) or Sum Col Norms (L21)
            total_L_rec_loss = 0.0
            
            with torch.no_grad():
                for x, _, indices in loader:
                    x = x.to(device)
                    indices = indices.to(device)
                    
                    S_old = self.S_memory[indices]
                    L_D_input = x - S_old
                    
                    # [Step 3]: L_D = D(E(L_D))
                    L_D_output = self(L_D_input)
                    
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
            c1 = (constraint_error ** 0.5) / (X_frobenius + 1e-9)
            c2 = (convergence_gap ** 0.5) / (X_frobenius + 1e-9)
            if c1 < tol or c2 < tol:
                print(f"\nConverged! c1={c1:.2e}, c2={c2:.2e}")
                break
            
            # Compute Averages and Objective for Reporting                    
            sparsity_term = total_sparsity_sum / total_pixels
            avg_L_rec_loss = total_L_rec_loss / total_pixels
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
            
            
        print(f"Final Reconstruction Loss: {avg_ae_loss:.4e}")      
        print("ADMM Training Complete.")