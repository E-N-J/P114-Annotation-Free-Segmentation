import torch
import torchvision.transforms.functional as TF
from PIL import Image

class UniversalNormalise(object):
    """
    Smart replacement for `transforms.ToTensor()` that also handles normalized tensors.

    - If fed a PIL Image, it scales standard pixels (0-255) to [0.0, 1.0].
    - If fed a z-scored Tensor, it applies windowing (-4 to +4) and maps to [0.0, 1.0].
    """
    def __init__(self, min_bound=-4.0, max_bound=4.0):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.range_val = max_bound - min_bound

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            clipped = torch.clamp(pic, min=self.min_bound, max=self.max_bound)
            scaled = (clipped - self.min_bound) / self.range_val
            return scaled
            
        elif isinstance(pic, Image.Image):
            return TF.to_tensor(pic)
            
        else:
            raise TypeError(f"Unexpected type provided to UniversalNormalise: {type(pic)}")