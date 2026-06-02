import torch
import torch.nn as nn
import math
from contextlib import contextmanager

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2] 
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class RDDPM(nn.Module):
    """
    Robust Denoising Diffusion Probabilistic Model (RDDPM)
    """
    def __init__(self, img_ch=1, base_ch=64, time_emb_dim=256, timesteps=1000, corrupt_ratio=0.25):
        super().__init__()
        self.timesteps = timesteps
        self.corrupt_ratio = corrupt_ratio
        
        beta_start = 0.001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # U-Net Architecture
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(img_ch, base_ch, 3, padding=1)
        
        self.downs = nn.ModuleList([
            Block(base_ch, base_ch * 2, time_emb_dim),
            Block(base_ch * 2, base_ch * 4, time_emb_dim),
            Block(base_ch * 4, base_ch * 8, time_emb_dim)
        ])
        
        self.bottleneck_conv = nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1)
        
        self.ups = nn.ModuleList([
            Block(base_ch * 8, base_ch * 4, time_emb_dim, up=True),
            Block(base_ch * 4, base_ch * 2, time_emb_dim, up=True),
            Block(base_ch * 2, base_ch, time_emb_dim, up=True)
        ])
        
        self.output = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    # Internal helpers to handle the 0 to 1 scaling issue
    def _scale_to_minus_one_to_one(self, x):
        return x * 2.0 - 1.0

    def _scale_to_zero_to_one(self, x):
        return (x + 1.0) / 2.0

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
            
        x = self.bottleneck_conv(x)
        
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1) 
            x = up(x, t)
            
        return self.output(x)

    @contextmanager
    def anomaly_generator(self, corrupt_ratio=None):
        is_training = self.training
        self.eval()
        
        ratio = corrupt_ratio if corrupt_ratio is not None else self.corrupt_ratio
        max_step = int(self.timesteps * ratio)

        def process_anom(batch_x):
            x_in = self._scale_to_minus_one_to_one(batch_x)
            B = x_in.shape[0]
            device = x_in.device

            with torch.no_grad():
                # Forward diffusion process
                t_max = torch.full((B,), max_step - 1, device=device, dtype=torch.long)
                noise = torch.randn_like(x_in)
                
                sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t_max])[:, None, None, None]
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t_max])[:, None, None, None]
                x_t = sqrt_alphas_cumprod * x_in + sqrt_one_minus_alphas_cumprod * noise

                # Iterative reverse denoising
                for i in reversed(range(0, max_step)):
                    t = torch.full((B,), i, device=device, dtype=torch.long)
                    predicted_noise = self(x_t, t)
                    
                    alpha_t = self.alphas[i]
                    alpha_cumprod_t = self.alphas_cumprod[i]
                    
                    if i > 0:
                        z = torch.randn_like(x_t)
                    else:
                        z = torch.zeros_like(x_t)
                        
                    x_t = (1 / torch.sqrt(alpha_t)) * (
                        x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
                    )
                    x_t = x_t + torch.sqrt(self.betas[i]) * z

                # Scale reconstruction back to [0, 1] range 
                recon_x = self._scale_to_zero_to_one(x_t)
                recon_x = torch.clamp(recon_x, 0.0, 1.0)
                
                # Calculate final anomaly map in the original input scale
                anom_map = torch.abs(batch_x - recon_x)
                
            return recon_x.detach(), anom_map.detach()

        try:
            yield process_anom
        finally:
            if is_training:
                self.train()