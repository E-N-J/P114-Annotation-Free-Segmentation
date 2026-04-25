import torch
import torch.nn as nn
from contextlib import contextmanager

class RobustVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, input_shape=(128, 128)):
        super(RobustVAE, self).__init__()

        self.input_h, self.input_w = input_shape

        if self.input_h % 8 != 0 or self.input_w % 8 != 0:
            raise ValueError("input_shape must be a multiple of 8 (e.g., 64, 128, 256) to ensure proper downsampling and upsampling.")
        
        self.feature_h = self.input_h // 8
        self.feature_w = self.input_w // 8
        self.flattened_dim = 256 * self.feature_h * self.feature_w
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_dim),
            nn.BatchNorm1d(self.flattened_dim),
            nn.ReLU(),
            nn.Unflatten(1, (256, self.feature_h, self.feature_w)),
            
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, in_channels, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar
    
    @contextmanager
    def anomaly_generator(self):
        self.eval()

        def process_anom(batch_x):
            with torch.no_grad():
                recon_x, _, _ = self(batch_x)
                anom_map = (batch_x - recon_x) ** 2
                
            return recon_x.detach(), anom_map.detach()

        yield process_anom