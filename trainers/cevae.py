import torch
import torch.optim as optim
from trainers.base import BaseTrainer
from pytorch_msssim import ssim

class CeVAETrainer(BaseTrainer):
    """
    Trainer for the Context-encoding Variational Autoencoder (ceVAE).
    Minimises the combined VAE and Context-Encoding (CE) objective.
    """
    def __init__(self, model, loader):
        super().__init__(model, loader)

    def _apply_ce_noise(self, x):
        """
        Applies Context-Encoding noise by masking 1-3 random spatial patches
        and filling them with values sampled from the batch's data distribution.
        """
        perturbed_x = x.clone()
        b, c, h, w = x.shape
        
        # Estimate the data distribution from the current batch
        batch_mean = x.mean().item()
        batch_std = x.std().item()
        
        for i in range(b):
            num_squares = torch.randint(1, 4, (1,)).item() 
            
            for _ in range(num_squares):
                # Randomly size the square
                mask_h = torch.randint(int(h * 0.1), int(h * 0.3) + 1, (1,)).item()
                mask_w = torch.randint(int(w * 0.1), int(w * 0.3) + 1, (1,)).item()
                
                # Random top-left corner
                top = torch.randint(0, h - mask_h + 1, (1,)).item()
                left = torch.randint(0, w - mask_w + 1, (1,)).item()
                
                # Fill with a random value from the data distribution
                noise = torch.empty((c, mask_h, mask_w), device=x.device).normal_(batch_mean, batch_std)
                noise.clamp_(0.0, 1.0)
                
                perturbed_x[i, :, top:top+mask_h, left:left+mask_w] = noise
                
        return perturbed_x

    def fit(self, epochs=60, lr=2e-4, lambda_=0.5):
        """
        Trains the ceVAE model.
        lambda_ balances the VAE objective against the CE objective (default 0.5).
        """
        print(f"\nTraining ceVAE Model...")
        
        optimiser = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        outer_pbar = self.tqdm(range(epochs), desc="Training", leave=True)
        
        for epoch in outer_pbar:
            total_loss = 0.0
            total_kl = 0.0
            total_vae_rec = 0.0
            total_ce_rec = 0.0
            
            for batch in self.loader: 
                x = batch[0].to(self.device)
                
                # Generate CE-noise perturbed data
                x_perturbed = self._apply_ce_noise(x).to(self.device)
                
                optimiser.zero_grad()
                
                # Standard VAE Pass
                recon_x_vae, mu, logvar = self.model(x, ce_mode=False)
                
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                ssim_val = ssim(x, recon_x_vae, data_range=1.0, size_average=True)
                rec_vae = (1 - ssim_val) * (x.shape[1] * x.shape[2] * x.shape[3])
                
                # Context-Encoding Pass
                recon_x_ce, _, _ = self.model(x_perturbed, ce_mode=True)
                
                ssim_ce_val = ssim(x, recon_x_ce, data_range=1.0, size_average=True)
                rec_ce = (1 - ssim_ce_val) * (x.shape[1] * x.shape[2] * x.shape[3])
                
                loss = (1 - lambda_) * (kl_loss + rec_vae) + (lambda_ * rec_ce)
                
                loss.backward()
                optimiser.step()
                
                # Accumulate metrics
                batch_size = x.size(0)
                total_loss += loss.item() / batch_size
                total_kl += kl_loss.item() / batch_size
                total_vae_rec += rec_vae.item() / batch_size
                total_ce_rec += rec_ce.item() / batch_size

            # Calculate epoch averages
            num_batches = len(self.loader)
            avg_loss = total_loss / num_batches
            avg_kl = total_kl / num_batches
            avg_vae_rec = total_vae_rec / num_batches
            avg_ce_rec = total_ce_rec / num_batches

            # Store in histories
            self.histories['loss'] = self.histories.get('loss', []) + [avg_loss]
            self.histories['kl_loss'] = self.histories.get('kl_loss', []) + [avg_kl]
            self.histories['vae_rec'] = self.histories.get('vae_rec', []) + [avg_vae_rec]
            self.histories['ce_rec'] = self.histories.get('ce_rec', []) + [avg_ce_rec]
            
            outer_pbar.set_postfix({
                'Avg Loss': f"{avg_loss:.1f}",
                'KL': f"{avg_kl:.1f}",
                'VAE Recon': f"{avg_vae_rec:.1f}",
                'CE Recon': f"{avg_ce_rec:.1f}"
            })
            
            if self.is_notebook:
                self.plot_metrics(log_scale=True)
                
        print(f"Training Complete. Final Avg Loss: {avg_loss:.4f}")
        self.log_final_metrics()    