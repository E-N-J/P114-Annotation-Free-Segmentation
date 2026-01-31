
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
    
