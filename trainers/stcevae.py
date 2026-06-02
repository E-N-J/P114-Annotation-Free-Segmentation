import torch
import torch.optim as optim
from models.stcevae import SpatialCeVAE
from trainers.base import BaseTrainer
from pytorch_msssim import ssim
import torch.nn.functional as F

class SpatialCeVAETrainer(BaseTrainer):
    """
    Trainer for the Spatial Context-encoding Variational Autoencoder (SpatialCeVAE).
    Minimises the combined VAE and Context-Encoding (CE) objective in the aligned space.
    """
    def __init__(self, model, loader, val_loader=None):
        super().__init__(model, loader, val_loader)

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

   

    def fit(self, epochs=60, warmup_epochs=10, lr=2e-4, stn_lr_mul=1, lambda_=0.5, beta=1.0):
        """
        Trains the SpatialCeVAE model.
        """
        print(f"\nTraining Spatial ceVAE Model...")
        
        stn_params = list(self.model.localisation.parameters()) + [self.model.global_offset]
        cevae_params = list(self.model.cevae.parameters())

        optimiser = optim.Adam([
            {'params': stn_params, 'lr': lr * stn_lr_mul}, 
            {'params': cevae_params, 'lr': lr}
        ])

        outer_pbar = self.tqdm(range(epochs), desc="Training", leave=True)
        
        for epoch in outer_pbar:
            self.model.train()
            
            total_loss = 0.0
            total_kl = 0.0
            total_vae_rec = 0.0
            total_ce_rec = 0.0
            total_congeal = 0.0

            is_post_warmup = epoch >= warmup_epochs
            
            for batch in self.loader: 
                x = batch[0].to(self.device)
                x_perturbed = self._apply_ce_noise(x).to(self.device)
                
                optimiser.zero_grad()
                
                # Standard VAE Pass (requesting aligned reconstruction)
                x_stn, recon_stn_vae, mu, logvar = self.model(
                    x, ce_mode=False, return_aligned=True, post_warmup=is_post_warmup
                ) 
                
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                
                ssim_val = ssim(x_stn, recon_stn_vae, data_range=1.0, size_average=True)
                rec_vae = (1 - ssim_val) * (x.shape[1] * x.shape[2] * x.shape[3])
                
                # Context-Encoding Pass (requesting aligned reconstruction)
                _, recon_stn_ce, _, _ = self.model(x_perturbed, ce_mode=True, return_aligned=True, post_warmup=is_post_warmup)
                
                ssim_ce_val = ssim(x_stn, recon_stn_ce, data_range=1.0, size_average=True)
                rec_ce = (1 - ssim_ce_val) * (x.shape[1] * x.shape[2] * x.shape[3])
                
                # Choose loss based on training phase
                loss = (1 - lambda_) * ((kl_loss * beta) + rec_vae) + (lambda_ * rec_ce)
                congealing_loss = x_stn.var(dim=0).sum()

                if not is_post_warmup:
                    congealing_loss.backward()
                else:
                    loss.backward()
             
                optimiser.step()
                
                # Accumulate metrics
                #batch_size = x.size(0)
                total_loss += loss.item() #/ batch_size
                total_kl += kl_loss.item() #/ batch_size
                total_vae_rec += rec_vae.item() #/ batch_size
                total_ce_rec += rec_ce.item() #/ batch_size
                total_congeal += congealing_loss.item() #/ batch_size

            # Calculate epoch averages
            num_batches = len(self.loader)
            avg_loss = total_loss / num_batches
            avg_kl = total_kl / num_batches
            avg_vae_rec = total_vae_rec / num_batches
            avg_ce_rec = total_ce_rec / num_batches
            avg_congeal = total_congeal / num_batches
            
            # Val loop
            avg_val_loss = 0.0
            # avg_val_congeal = 0.0
            
            if hasattr(self, 'val_loader') and self.val_loader is not None:
                self.model.eval()
                val_total_loss = 0.0
                # val_total_congeal = 0.0
                
                with torch.no_grad():
                    for val_batch in self.val_loader:
                        x_val = val_batch[0].to(self.device)
                        x_val_perturbed = self._apply_ce_noise(x_val).to(self.device)

                        x_stn_val, recon_stn_vae_val, mu_val, logvar_val = self.model(
                            x_val, ce_mode=False, return_aligned=True, post_warmup=is_post_warmup
                        )
                        _, recon_stn_ce_val, _, _ = self.model(
                            x_val_perturbed, ce_mode=True, return_aligned=True, post_warmup=is_post_warmup
                        )

                        kl_loss_val = -0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp(), dim=1).mean()
                        ssim_val_val = ssim(x_stn_val, recon_stn_vae_val, data_range=1.0, size_average=True)
                        rec_vae_val = (1 - ssim_val_val) * (x_val.shape[1] * x_val.shape[2] * x_val.shape[3])
                        
                        ssim_ce_val_val = ssim(x_stn_val, recon_stn_ce_val, data_range=1.0, size_average=True)
                        rec_ce_val = (1 - ssim_ce_val_val) * (x_val.shape[1] * x_val.shape[2] * x_val.shape[3])

                        loss_val = (1 - lambda_) * (kl_loss_val * beta + rec_vae_val) + (lambda_ * rec_ce_val)
                        # congeal_loss_val = x_stn_val.var(dim=0).mean()

                        #batch_size_val = x_val.size(0)
                        val_total_loss += loss_val.item() #/ batch_size_val
                        # val_total_congeal += congeal_loss_val.item() #/ batch_size_val
                
                num_val_batches = len(self.val_loader)
                avg_val_loss = val_total_loss / num_val_batches
                # avg_val_congeal = val_total_congeal / num_val_batches
            
            # Store in histories
            # self.histories['loss'] = self.histories.get('loss', []) + [avg_loss]
            # self.histories['kl_loss'] = self.histories.get('kl_loss', []) + [avg_kl]
            # self.histories['vae_rec'] = self.histories.get('vae_rec', []) + [avg_vae_rec]
            # self.histories['ce_rec'] = self.histories.get('ce_rec', []) + [avg_ce_rec]
            self.histories['congealing_loss'] = self.histories.get('congealing_loss', []) + [avg_congeal]
            
            # Save validation to histories without altering the progress bar
            if hasattr(self, 'val_loader') and self.val_loader is not None:
                # self.histories['val_loss'] = self.histories.get('val_loss', []) + [avg_val_loss]
                # self.histories['val_congealing_loss'] = self.histories.get('val_congealing_loss', []) + [avg_val_congeal]
                pass
            
            outer_pbar.set_postfix({
                'Phase': 'ceVAE' if is_post_warmup else 'STN Align',
                'Loss': f"{avg_loss:.1f}",
                'Congl': f"{avg_congeal:.9f}",
                'VAE Rec': f"{avg_vae_rec:.1f}",
                'CE Rec': f"{avg_ce_rec:.1f}",
                'KL': f"{avg_kl:.1f}",
            })
            
            if self.is_notebook:
                self.plot_metrics(log_scale=True)
                
        print(f"Training Complete. Final Avg Loss: {avg_loss:.4f}" + 
              (f" | Final Val Loss: {avg_val_loss:.4f}" if hasattr(self, 'val_loader') and self.val_loader else ""))
        self.log_final_metrics()