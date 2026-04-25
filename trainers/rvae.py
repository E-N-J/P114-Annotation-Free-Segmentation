import torch
import torch.optim as optim
from trainers.base import BaseTrainer
class RVAETrainer(BaseTrainer): # TODO: include references to all relevant papers 
    def __init__(self, model, loader):
        super().__init__(model, loader)

    def fit(self, epochs=60, lr=1e-3, beta=0.005):
        """
        Trains the RVAE model using beta-divergence loss.
        beta controls the robustness to outliers.
        """
        print(f"\nTraining RVAE Model...")
        
        optimiser = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        outer_pbar = self.tqdm(range(epochs), desc="Training", leave=True)
        
        for epoch in outer_pbar:
            total_loss = 0.0
            total_kl = 0.0
            total_beta_rec = 0.0
            total_l1_loss = 0.0
            
            for batch in self.loader: 
                x = batch[0].to(self.device)
                
                optimiser.zero_grad()
                
                recon_x, mu, logvar = self.model(x)
                
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                
                sse = torch.sum((recon_x - x) ** 2, dim=list(range(1, x.dim()))) # Sum of squared errors per sample
                D = x[0].numel() # Total number of features/pixels per image
                
                log_c_norm = - (beta * D / 2.0) * torch.log(torch.tensor(2 * torch.pi, device=self.device))
                c_norm = torch.exp(log_c_norm)
                
                beta_loss = - ((beta + 1) / beta) * c_norm * torch.exp(- (beta / 2.0) * sse)
                
                loss = torch.mean(beta_loss + kl_div)
                
                loss.backward()
                optimiser.step()

                l1_loss = torch.mean(torch.abs(recon_x - x)).item()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_kl += torch.mean(kl_div).item()
                total_beta_rec += torch.mean(beta_loss).item()
                total_l1_loss += l1_loss

            # Calculate epoch averages
            num_batches = len(self.loader)
            avg_loss = total_loss / num_batches
            avg_kl = total_kl / num_batches
            avg_beta_rec = total_beta_rec / num_batches
            avg_l1_loss = total_l1_loss / num_batches

            # Store in histories
            self.histories['loss'] = self.histories.get('loss', []) + [avg_loss]
            # self.histories['kl_loss'] = self.histories.get('kl_loss', []) + [avg_kl]
            self.histories['beta_rec'] = self.histories.get('beta_rec', []) + [avg_beta_rec]
            # self.histories['l1_loss'] = self.histories.get('l1_loss', []) + [avg_l1_loss]
            
            outer_pbar.set_postfix({
                'Avg Loss': f"{avg_loss:.4f}",
                'KL': f"{avg_kl:.2f}",
                'Beta Recon': f"{avg_beta_rec:.4f}",
                'L1 Loss': f"{avg_l1_loss:.4f}"
            })
            
            if self.is_notebook:
                self.plot_metrics(log_scale=True) 
                
        print(f"Training Complete. Final Avg Loss: {avg_loss:.4f}")
        self.log_final_metrics()