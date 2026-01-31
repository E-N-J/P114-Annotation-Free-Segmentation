import torch
import torch.nn as nn
from .utils import GaussianNoise

class RobustDeepAutoencoder(nn.Module):
    """
    Robust Deep Autoencoder for decomposing images into low-rank and sparse components.
    Inspired by Zhou & Paffenroth, Anomaly Detection with Robust Deep Autoencoders.
    """
    def __init__(self, latent_dim=3, dropout=0.1, std=0.1):
        super().__init__()

        self.loss = nn.MSELoss()
        
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

    def forward(self, x):
        x = self.conv_encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        x = self.linear_encoder(x)
        x = self.linear_decoder(x)
        
        x = x.view(-1, 64, 42, 42)
        L_pred = self.conv_decoder(x)
        
        return L_pred