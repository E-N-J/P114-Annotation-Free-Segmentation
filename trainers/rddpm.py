import torch.optim as optim
import torch.nn.functional as F
import torch
from .base import BaseTrainer

def robust_huber_loss(pred, target, delta=0.2):
    return F.huber_loss(pred, target, delta=delta, reduction='mean')

def robust_lts_loss(pred, target, lambda_=0.8):
    pixel_loss = F.mse_loss(pred, target, reduction='none')
    sample_loss = pixel_loss.mean(dim=[1, 2, 3])
    
    B = sample_loss.shape[0]
    s = max(1, int(lambda_ * B)) 

    trimmed_loss, _ = torch.topk(sample_loss, s, largest=False)
    
    return trimmed_loss.mean()

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=0.001, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
    
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t, noise):
        """Forward diffusion process: closed form"""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
    
class RDDPMTrainer(BaseTrainer):
    def __init__(self, model, loader, timesteps=1000):
        super().__init__(model, loader)
            
        self.scheduler = NoiseScheduler(timesteps=timesteps, device=self.device)
        
    def fit(self, lr=1e-4, epochs=20, loss_type='huber', robust_param=None):
        self.loss_type = loss_type
        if self.loss_type == 'huber':
            self.delta = robust_param if robust_param is not None else 0.2
        elif self.loss_type == 'lts':
            self.lambda_ = robust_param if robust_param is not None else 0.8
        print(f"\nTraining RDDPM with {self.loss_type.upper()} loss...")
        
        optimiser = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        outer_pbar = self.tqdm(range(epochs), desc="Epochs", leave=True)
        
        for epoch in outer_pbar:
            total_loss = 0.0
            
            inner_pbar = self.tqdm(self.loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in inner_pbar:
                # Handle varying dataloader outputs (adjust index based on your framework)
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                x = self.model._scale_to_minus_one_to_one(x)
                B = x.shape[0]
                
                # Sample uniform random timesteps
                t = torch.randint(0, self.scheduler.timesteps, (B,), device=self.device).long()
                
                # Sample standard Gaussian noise
                noise = torch.randn_like(x)
                
                # Add noise to the clean images
                x_noisy = self.scheduler.add_noise(x, t, noise)
                
                # Predict the noise
                optimiser.zero_grad()
                predicted_noise = self.model(x_noisy, t)
                
                # Calculate  loss
                if self.loss_type == 'huber':
                    loss = robust_huber_loss(predicted_noise, noise, delta=self.delta)
                else: # 'lts'
                    loss = robust_lts_loss(predicted_noise, noise, lambda_=self.lambda_)
                    
                loss.backward()
                optimiser.step()
                
                total_loss += loss.item()
                inner_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
            avg_loss = total_loss / len(self.loader)
            self.histories['rddpm_loss'] = self.histories.get('rddpm_loss', []) + [avg_loss]
            outer_pbar.set_postfix({'Avg Loss': f"{avg_loss:.4f}"})
            
            if self.is_notebook:
                self.plot_metrics(log_scale=True)
                
        print("RDDPM Training Complete.")
        self.log_final_metrics()

    @torch.no_grad()
    def reconstruct_anomalies(self, x_anomalous, corrupt_ratio=0.25):
        """
        Inference step for anomaly segmentation.
        Adds noise up to corrupt_ratio (e.g., 25% of timesteps) and denoises.
        """
        self.model.eval()
        B = x_anomalous.shape[0]
        max_step = int(self.scheduler.timesteps * corrupt_ratio)
        
        # Forward process to max_step
        t_max = torch.full((B,), max_step - 1, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_anomalous)
        x_t = self.scheduler.add_noise(x_anomalous, t_max, noise)
        
        # Iterative backward denoising
        for i in reversed(range(0, max_step)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x_t, t)
            
            alpha_t = self.scheduler.alphas[i]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[i]
            
            # Reparameterisation trick for backward step
            if i > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
                
            # Compute x_{t-1}
            x_t = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )
            # Simplified variance for the backward step
            beta_t = self.scheduler.betas[i]
            x_t = x_t + torch.sqrt(beta_t) * z
            
        # The anomaly heatmap is the absolute difference
        anomaly_heatmap = torch.abs(x_anomalous - x_t)
        
        return x_t, anomaly_heatmap