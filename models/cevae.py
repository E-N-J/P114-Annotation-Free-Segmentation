import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from torchvision.transforms.functional import gaussian_blur
from captum.attr import GuidedBackprop, NoiseTunnel
from contextlib import contextmanager

from .utils import CoordConv, CoordConvTranspose, CeVAEInferenceLoss

class ContextEncodingVAE(nn.Module):
    """
    Strictly Fully Convolutional Context-encoding VAE matching Zimmerer et al. (2019).
    """
    def __init__(self, latent_channels=1024, with_r=False):
        super().__init__()
        
        self.encoder = nn.Sequential(
            CoordConv(1, 16, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConv(16, 64, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConv(64, 256, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConv(256, 1024, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConv(1024, 1024, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_mu = CoordConv(1024, latent_channels, kernel_size=1)
        self.conv_logvar = CoordConv(1024, latent_channels, kernel_size=1)
        
        self.decoder_input = CoordConv(latent_channels, 1024, kernel_size=1)
        
        self.decoder = nn.Sequential(
            CoordConvTranspose(1024, 1024, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConvTranspose(1024, 256, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConvTranspose(256, 64, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConvTranspose(64, 16, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.LeakyReLU(0.2, inplace=True),
            
            CoordConvTranspose(16, 1, kernel_size=4, stride=2, padding=1, with_r=with_r),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        """Applies the reparameterisation trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x, ce_mode=False):
        """
        Forward pass with distinct routing for the Context-Encoding (CE) task.
        """
        mu, logvar = self.encode(x)
        
        if ce_mode:
            z = mu 
        else:
            z = self.reparameterise(mu, logvar)
            
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar

    @contextmanager
    def anomaly_generator(
        self, 
        nt_samples=50, 
        nt_samples_batch_size=10,
    ):
        """
        Context manager that sets up Captum attribution models once, yields a 
        batch-processing function, and safely tears down memory afterwards.
        """
        
        tmp_model = copy.deepcopy(self)
        
        def replace_layers(module, find=nn.LeakyReLU, replace=nn.ReLU):
            for child_name, child in module.named_children():
                if isinstance(child, find):
                    setattr(module, child_name, replace())
                else:
                    replace_layers(child, find, replace)

        replace_layers(tmp_model)
            
        tmp_model.eval()

        lw = CeVAEInferenceLoss(tmp_model)
        bp = GuidedBackprop(lw)
        nt = NoiseTunnel(bp)

        def process_anom(batch_x):
            x_in = batch_x.clone().detach().requires_grad_(True)
            with torch.no_grad():
                recon_x, _, _ = self(batch_x, ce_mode=False)
                rec_error = F.l1_loss(batch_x, recon_x, reduction='none')
         
            with torch.enable_grad():
                grad = nt.attribute(
                    x_in, 
                    nt_samples=nt_samples, 
                    nt_type='smoothgrad',
                    stdevs=0.3, 
                    nt_samples_batch_size=nt_samples_batch_size,
                )
                
            grad = gaussian_blur(torch.abs(grad), kernel_size=[5, 5], sigma=[1.0, 1.0])
            anom_map = rec_error * grad
            
            return recon_x.detach(), anom_map.detach() 

        yield process_anom
            
        