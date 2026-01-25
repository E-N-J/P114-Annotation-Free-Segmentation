import torch

def shrinkage_l1(x, lamb):
    """ Element-wise Soft Thresholding. """
    # S = sign(x) * max(|x| - lambda, 0)
    return torch.sign(x) * torch.maximum(
        torch.abs(x) - lamb, 
        torch.tensor(0.0, device=x.device)
    )

def shrinkage_l21(S, lamb):
    """ Group-wise Soft Thresholding. Groups = Columns. """
    # Calculate L2 norm of each feature (column) across the batch
    col_norms = torch.norm(S, p=2, dim=0, keepdim=True) + 1e-10
    
    # Scale factor = max(1 - lambda/norm, 0)
    scale_factor = torch.maximum(
        1 - (lamb / col_norms),
        torch.tensor(0.0, device=S.device)
    )
    return S * scale_factor
