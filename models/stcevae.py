import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from captum.attr import GuidedBackprop, NoiseTunnel

from .cevae import ContextEncodingVAE
from .utils import CeVAEInferenceLoss


class SpatialCeVAE(nn.Module):
    """Spatial Transformer Network (STN) composed with Context-Encoding VAE.

    The localization network predicts an affine transform, the input is warped
    into a aligned frame, and the ceVAE reconstructs that aligned image.
    Reconstructions are then inverse-warped back to the original frame.
    """

    def __init__(
        self,
        latent_channels=1024,
        with_r=True,
        input_shape=(128, 128),
    ):
        super().__init__()

        self.input_shape = input_shape
        self.cevae = ContextEncodingVAE(latent_channels=latent_channels, with_r=with_r, input_shape=input_shape)

        self.localisation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.localisation[-1].weight)
        nn.init.zeros_(self.localisation[-1].bias)

        self.global_offset = nn.Parameter(torch.zeros(3))

    def _predict_theta(self, x, post_warmup=False):
        params = self.localisation(x)
        if post_warmup:
            # Detach freezes the CNN's relative alignments.
            # Adding global_offset attaches it to the graph to receive all downstream gradients.
            params = params.detach()

        # Bound the angle using tanh. 
        angle = torch.tanh(params[:, 0]) * torch.pi 
        
        # Bound the translations.
        # 0.2 means it can only shift the image by a maximum of 20% in any direction.
        tx = torch.tanh(params[:, 1]) * 0
        ty = torch.tanh(params[:, 2]) * 0

        if post_warmup:
            angle = angle + self.global_offset[0]
            tx = tx + self.global_offset[1]
            ty = ty + self.global_offset[2]

        # Calculate trigonometry
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Construct the rigid transformation matrix
        theta = torch.stack([
            torch.stack([cos_a, -sin_a, tx], dim=-1),
            torch.stack([sin_a, cos_a, ty], dim=-1)
        ], dim=-2)
        
        return theta
    
    def _warp(self, x, theta, is_image=True):
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        warped = F.grid_sample(x, grid, mode='bicubic', align_corners=False, padding_mode="zeros")
        
        if is_image:
            return torch.clamp(warped, 0.0, 1.0)
        return warped

    def _invert_theta(self, theta):
        batch_size = theta.size(0)
        homogeneous = torch.zeros(batch_size, 3, 3, device=theta.device, dtype=theta.dtype)
        homogeneous[:, :2, :] = theta
        homogeneous[:, 2, 2] = 1.0
        inverse = torch.linalg.inv(homogeneous)
        return inverse[:, :2, :]

    def forward(self, x, ce_mode=False, return_aligned=False, post_warmup=False):
        """
        Forward pass with a flag to return aligned tensors for training.
        """
        theta = self._predict_theta(x, post_warmup=post_warmup)
        x_stn = self._warp(x, theta)

        recon_stn, mu, logvar = self.cevae(x_stn, ce_mode=ce_mode)
        
        # Return aligned tensors during training to prevent STN collapse
        if return_aligned:
            return x_stn, recon_stn, mu, logvar

        # Otherwise, inverse warp back to original space (for inference)
        inverse_theta = self._invert_theta(theta)
        recon_x = self._warp(recon_stn, inverse_theta)

        return recon_x, mu, logvar

    @contextmanager
    def anomaly_generator(self, nt_samples=50, nt_samples_batch_size=10):
        def process_anom(batch_x):
            with torch.no_grad():
                theta = self._predict_theta(batch_x).detach()
                x_stn = self._warp(batch_x, theta)
                recon_stn, _, _ = self.cevae(x_stn, ce_mode=False)
                inverse_theta = self._invert_theta(theta)
                rec_error = F.l1_loss(x_stn, recon_stn, reduction='none')

            lw = CeVAEInferenceLoss(copy.deepcopy(self.cevae))
            lw.eval()
            bp = GuidedBackprop(lw)
            nt = NoiseTunnel(bp)

            x_in = x_stn.clone().detach().requires_grad_(True)

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

            recon_x = self._warp(recon_stn, inverse_theta, is_image=True)
            anom_map = self._warp(anom_map, inverse_theta, is_image=False)

            return recon_x.detach(), anom_map.detach()

        yield process_anom