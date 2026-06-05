import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBottleneck(nn.Module):
    def __init__(self, flatten_dim, hidden_dim=1024, latent_dim=32, dropout_p=0.2, noise_std=0.1, tied_weights=True):
        super().__init__()
        self.tied_weights = tied_weights
        
        # Define the encoder weights 
        self.weight_1 = nn.Parameter(torch.randn(hidden_dim, flatten_dim) / flatten_dim**0.5)
        self.weight_2 = nn.Parameter(torch.randn(latent_dim, hidden_dim) / hidden_dim**0.5)
        
        # Define independent decoder weights (only if untied)
        if not self.tied_weights:
            self.weight_d1 = nn.Parameter(torch.randn(hidden_dim, latent_dim) / latent_dim**0.5)
            self.weight_d2 = nn.Parameter(torch.randn(flatten_dim, hidden_dim) / hidden_dim**0.5)
        
        # Define the independent biases (always independent)
        self.bias_e1 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_e2 = nn.Parameter(torch.zeros(latent_dim))
        self.bias_d1 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_d2 = nn.Parameter(torch.zeros(flatten_dim))

        # Define regularisation components
        self.dropout = nn.Dropout(p=dropout_p)
        self.noise_std = noise_std

        self.bn_e1 = nn.BatchNorm1d(hidden_dim)
        self.bn_d1 = nn.BatchNorm1d(hidden_dim)

    def encode(self, x):
        h1 = F.linear(x, self.weight_1, self.bias_e1)
        h1 = F.leaky_relu(self.bn_e1(h1), 0.2)
        h1 = self.dropout(h1)
        
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(h1) * self.noise_std
            h1 = h1 + noise
            
        h2 = F.linear(h1, self.weight_2, self.bias_e2)
        return h2

    def decode(self, h):
        if self.tied_weights:
            w1 = self.weight_2.t()
            w2 = self.weight_1.t()
        else:
            w1 = self.weight_d1
            w2 = self.weight_d2

        d1 = F.linear(h, w1, self.bias_d1)
        d1 = F.leaky_relu(self.bn_d1(d1), 0.2)
        d2 = F.linear(d1, w2, self.bias_d2)
        return d2
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

class RobustDeepAutoencoder(nn.Module):
    def __init__(self, input_shape=(128, 128), latent_dim=1024, hidden_dim=1024, dropout=0.0, noise_std=0.0):
        super().__init__()
        
        self.input_h, self.input_w = input_shape
        self.feature_h = self.input_h // 16 
        self.feature_w = self.input_w // 16
        
        self.flatten_dim = 64 * self.feature_h * self.feature_w # 64 * 8 * 8 = 4096
        
        self.conv_encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        
        self.tied_bottleneck = LinearBottleneck(
            self.flatten_dim, 
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim,
            dropout_p=dropout, 
            noise_std=noise_std, 
            tied_weights=True
        )

        self.conv_decoder = nn.Sequential(
            # 8x8 -> 16x16 (Starting from 64 channels to match the reshape)
            nn.Conv2d(64, 256, kernel_size=3, padding=1), # 64 * 4 = 256
            nn.PixelShuffle(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Smoothing
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 32 * 4 = 128
            nn.PixelShuffle(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Smoothing
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16 * 4 = 64
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Smoothing
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 128x128
            nn.Conv2d(16, 4, kernel_size=3, padding=1),   # 1 * 4 = 4
            nn.PixelShuffle(2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),    # Smoothing
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Spatial Compression
        x = self.conv_encoder(x)
        
        # Flatten 
        x = torch.flatten(x, start_dim=1)
        
        # Tied-Weight Linear Bottleneck
        x = self.tied_bottleneck(x)
        # Unflatten 
        x_reshaped = x.view(-1, 64, self.feature_h, self.feature_w)
        
        # Spatial Expansion (PixelShuffle)
        reconstruction = self.conv_decoder(x_reshaped)
        
        return reconstruction
    
