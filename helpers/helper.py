
import torch
import torch.nn as nn

def robust_loss(x, L_pred, lambda_=0.1):
        """
        Loss = ||X - L - S|| + lambda * ||S||_1
        Assuming S = x - L (residual), we want to minimize L2 of reconstruction
        BUT allow for large sparse errors (L1 penalty on residual).
        """
        residual = x - L_pred
        l2_loss = torch.norm(residual, p=2)
        l1_loss = torch.norm(residual, p=1)
        
        loss = l2_loss + lambda_ * l1_loss
        return loss, l2_loss, l1_loss

def shrinkage_l1(x, lamb):
    """ Element-wise Soft Thresholding (Eq 11 in Paper). """
    # S = sign(x) * max(|x| - lambda, 0)
    return torch.sign(x) * torch.maximum(
        torch.abs(x) - lamb, 
        torch.tensor(0.0, device=x.device)
    )

def shrinkage_l21(S, lamb):
    """ Group-wise Soft Thresholding (Eq 12 in Paper). Groups = Columns. """
    # Calculate L2 norm of each feature (column) across the batch
    col_norms = torch.norm(S, p=2, dim=0, keepdim=True) + 1e-10
    
    # Scale factor = max(1 - lambda/norm, 0)
    scale_factor = torch.maximum(
        1 - (lamb / col_norms),
        torch.tensor(0.0, device=S.device)
    )
    return S * scale_factor

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x
    
# class IndexedDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         data, _ = self.dataset[idx]
#         return data, idx
    
