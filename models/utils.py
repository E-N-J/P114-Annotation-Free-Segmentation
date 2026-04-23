
import torch
import torch.nn as nn
import torch.nn.functional as F
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x
    
class CeVAEInferenceLoss(nn.Module):
    def __init__(self, model, window_size=11, data_range=1.0):
        super().__init__()
        self.model = model
        self.window_size = window_size
        self.data_range = data_range
        
    def forward(self, x):
        recon_x, mu_val, logvar_val = self.model(x, ce_mode=False)
        
        kl_div = -0.5 * (1 + logvar_val - mu_val.pow(2) - logvar_val.exp())
        kl_div = kl_div.view(x.shape[0], -1).sum(dim=1)

        rec_loss = F.mse_loss(recon_x, x, reduction='none').view(x.shape[0], -1).sum(dim=1)
        
        return kl_div + rec_loss
class AddCoordinates(object):
    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    """
    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        # Optimisation: generate coordinates directly on the input image's device
        y_coords = 2.0 * torch.arange(image_height, device=image.device).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width, device=image.device).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        image = torch.cat((coords, image), dim=1)

        return image


class CoordConv(nn.Module): # TODO: add as a depemdency from the original project repo
    r"""2D Convolution Module Using Extra Coordinate Information"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False):
        super(CoordConv, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_layer = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size, 
            stride=stride,
            padding=padding, 
            dilation=dilation,
            groups=groups, 
            bias=bias
            )

        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)
        return x


class CoordConvTranspose(nn.Module):
    r"""2D Transposed Convolution Module Using Extra Coordinate Information"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, with_r=False):
        super(CoordConvTranspose, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_tr_layer = nn.ConvTranspose2d(
            in_channels, 
            out_channels,
            kernel_size, 
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups, 
            bias=bias,
            dilation=dilation)

        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_tr_layer(x)
        return x
