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

class RDDPMTrainer(BaseTrainer):
    def __init__(self, model, loader):
        super().__init__(model, loader)
        
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
                # Handle varying dataloader outputs
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                
                # Scale [0, 1] inputs to [-1, 1] for symmetric noise addition
                x = self.model._scale_to_minus_one_to_one(x)
                B = x.shape[0]
                
                # Sample uniform random timesteps natively from the model's attribute
                t = torch.randint(0, self.model.timesteps, (B,), device=self.device).long()
                
                # Sample standard Gaussian noise
                noise = torch.randn_like(x)
                
                # Forward diffusion process using the model's registered buffers
                sqrt_alphas_cumprod = torch.sqrt(self.model.alphas_cumprod[t])[:, None, None, None]
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.model.alphas_cumprod[t])[:, None, None, None]
                x_noisy = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
                
                # Predict the noise
                optimiser.zero_grad()
                predicted_noise = self.model(x_noisy, t)
                
                # Calculate robust loss
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