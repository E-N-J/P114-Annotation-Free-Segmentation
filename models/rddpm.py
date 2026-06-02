import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import contextmanager

class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes the discrete time step t into a continuous vector."""
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

class ResBlock(nn.Module):
    """Residual block with time embedding projection."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        time_emb = self.time_mlp(t)[..., None, None]
        h = h + time_emb
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Standard self-attention block used in the ablated U-Net."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(B, C, H, W)
        return x + self.proj(out)

class LDM_UNet(nn.Module):
    """The heavy 'ablated UNet' backbone utilised in the Latent Diffusion Models paper."""
    def __init__(self, img_channels=3, base_channels=128, channel_mults=(1, 2, 4, 4), time_emb_dim=512):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        # Downsampling path
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(2):
                # Group layers into a single block to ensure 1:1 skip connection mapping
                block = nn.ModuleList([ResBlock(now_channels, out_channels, time_emb_dim)])
                now_channels = out_channels
                # Apply attention at 32x32, 16x16, and 8x8 resolutions
                if i in [1, 2, 3]: 
                    block.append(AttentionBlock(now_channels))
                self.downs.append(block)
                channels.append(now_channels)
                
            if i != len(channel_mults) - 1:
                down_conv = nn.ModuleList([nn.Conv2d(now_channels, now_channels, kernel_size=3, stride=2, padding=1)])
                self.downs.append(down_conv)
                channels.append(now_channels)

        # Bottleneck
        self.mid_block1 = ResBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResBlock(now_channels, now_channels, time_emb_dim)

        # Upsampling path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(3):
                block = nn.ModuleList([ResBlock(now_channels + channels.pop(), out_channels, time_emb_dim)])
                now_channels = out_channels
                if i in [1, 2, 3]:
                    block.append(AttentionBlock(now_channels))
                self.ups.append(block)
                
            if i != 0:
                up_conv = nn.ModuleList([nn.ConvTranspose2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1)])
                self.ups.append(up_conv)

        self.final_norm = nn.GroupNorm(32, now_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(now_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        
        skips = [x]
        
        # Iterating through block groups ensures perfectly synchronised skips
        for block in self.downs:
            for layer in block:
                x = layer(x, t_emb) if isinstance(layer, ResBlock) else layer(x)
            skips.append(x)
            
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block in self.ups:
            if isinstance(block[0], ResBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            for layer in block:
                x = layer(x, t_emb) if isinstance(layer, ResBlock) else layer(x)

        x = self.final_act(self.final_norm(x))
        return self.final_conv(x)


class MicroLDM_UNet(nn.Module):
    """
    A hyper-optimised, lightweight version of the LDM U-Net.
    """
    def __init__(self, img_channels=1, base_channels=32, channel_mults=(1, 2, 4, 4), time_emb_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)

        # Strictly decouple ResBlocks from down/up convolutions
        self.downs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        channels = []
        
        now_channels = base_channels
        
        # Downsampling: 128 -> 64 -> 32 -> 16
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            self.downs.append(ResBlock(now_channels, out_channels, time_emb_dim))
            now_channels = out_channels
            channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.down_convs.append(nn.Conv2d(now_channels, now_channels, kernel_size=3, stride=2, padding=1))

        # Bottleneck (16x16 resolution)
        self.mid_block1 = ResBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResBlock(now_channels, now_channels, time_emb_dim)

        # Upsampling: 16 -> 32 -> 64 -> 128
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            skip_channels = channels.pop()
            self.ups.append(ResBlock(now_channels + skip_channels, out_channels, time_emb_dim))
            now_channels = out_channels
                
            if i != 0:
                self.up_convs.append(nn.ConvTranspose2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1))

        self.final_norm = nn.GroupNorm(8, now_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(now_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        
        skips = []
        
        # Encoder
        for i in range(len(self.downs)):
            x = self.downs[i](x, t_emb)
            skips.append(x) # Save perfectly aligned skip connection
            if i < len(self.down_convs):
                x = self.down_convs[i](x)
            
        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for i in range(len(self.ups)):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1) # Dimensions mathematically guaranteed to match
            x = self.ups[i](x, t_emb)
            if i < len(self.up_convs):
                x = self.up_convs[i](x)

        x = self.final_act(self.final_norm(x))
        return self.final_conv(x)

class RDDPM(nn.Module):
    """
    Robust Denoising Diffusion Probabilistic Model.
    Designed for integration with standard context-manager-based anomaly pipelines.
    """
    def __init__(self, img_channels=1, timesteps=1000, corrupt_ratio=0.25):
        super().__init__()
        self.timesteps = timesteps
        self.corrupt_ratio = corrupt_ratio
        
        self.unet = MicroLDM_UNet(img_channels=img_channels)
        
        # Noise Schedule
        beta_start = 0.001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def _scale_to_minus_one_to_one(self, x):
        return x * 2.0 - 1.0

    def _scale_to_zero_to_one(self, x):
        return (x + 1.0) / 2.0

    def forward(self, x, timestep):
        """Standard forward pass predicting noise."""
        return self.unet(x, timestep)

    @contextmanager
    def anomaly_generator(self, corrupt_ratio=None):
        """
        Context manager that yields a batch-processing function to generate
        reconstructions and anomaly maps.
        """
        is_training = self.training
        self.eval()
        
        ratio = corrupt_ratio if corrupt_ratio is not None else self.corrupt_ratio
        max_step = int(self.timesteps * ratio)

        def process_anom(batch_x):
            # Scale incoming [0, 1] data to [-1, 1]
            x_in = self._scale_to_minus_one_to_one(batch_x)
            B = x_in.shape[0]
            device = x_in.device

            with torch.no_grad():
                # Corrupt up to max_step
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

                # Rescale and compute heatmap
                recon_x = self._scale_to_zero_to_one(x_t)
                recon_x = torch.clamp(recon_x, 0.0, 1.0)
                anom_map = torch.abs(batch_x - recon_x)
                
            return recon_x.detach(), anom_map.detach()

        try:
            yield process_anom
        finally:
            if is_training:
                self.train()