import os
import glob
import torch
import torch.nn as nn
import kornia.augmentation as K_transforms
import kornia.geometry.transform as K_geometry
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from .flatDataset import FlatDataset 

class Augmentor:
    def __init__(self, source_root, rotation=0, tps=0, elastic=0, dense_noise=0, device='cuda'):
        """
        Args:
            source_root: Path to the flat folder of raw images.
            rotation: Degrees of random rotation.
            tps: Scale of Thin Plate Spline distortion.
            elastic: Scale of Elastic Transform distortion.
            dense_noise: Scale of dense noise.
            device: Device to run augmentations on.
            """
        self.source_root = source_root.rstrip(os.sep)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.is_identity = (rotation == 0 and tps == 0 and elastic == 0 and dense_noise == 0)

        if not self.is_identity:
            # Define the GPU Augmentation Pipeline
            augs = []
            if rotation > 0:
                augs.append(K_transforms.RandomRotation(degrees=rotation, p=1.0))
            if tps > 0:
                random_tps = create_tps_transform(strength=tps, device=self.device)
                augs.append(random_tps)
            if elastic > 0:
                augs.append(K_transforms.RandomElasticTransform(alpha=(elastic, elastic), p=1.0, sigma=(elastic*50, elastic*50)))
            if dense_noise > 0:
                augs.append(RandomRicianNoise(std=dense_noise))
                
            self.aug = transforms.Compose(augs)

            # Construct the destination folder name
            folder_name = os.path.basename(self.source_root)
            suffix = f"_Rot{rotation}_TPS{tps}_El{elastic}_DN{dense_noise}"
            parent_dir = os.path.dirname(self.source_root)
            self.dest_root = os.path.join(parent_dir, folder_name + suffix)
        else:
            # If no augmentation, the destination is just the source
            self.dest_root = self.source_root
            
        files = glob.glob(os.path.join(self.source_root, "*.*"))
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if valid_files:
            tmp_img = read_image(valid_files[0], mode=ImageReadMode.RGB)
            self.init_dims = (tmp_img.shape[1], tmp_img.shape[2]) # H, W

    def prepare(self, force_rebuild=False, batch_size=128):
        """
        Runs the augmentation pipeline in batches and saves images to disk.
        """
        if self.is_identity:
            print(f"Identity augmentation requested. Using raw source: {self.source_root}")
            return

        if os.path.exists(self.dest_root) and not force_rebuild:
            print(f"Augmented dataset found at: {self.dest_root}")
            return

        print(f"Generating augmented data to: {self.dest_root}")
        os.makedirs(self.dest_root, exist_ok=True)
        
        files = glob.glob(os.path.join(self.source_root, "*.*"))
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not valid_files:
            raise ValueError(f"No images found in {self.source_root}. Ensure it is a flat folder.")

        # Process the files in chunks defined by batch_size
        for i in tqdm(range(0, len(valid_files), batch_size), desc="Augmenting in Batches"):
            batch_files = valid_files[i : i + batch_size]
            img_list = []
            
            # Read a batch of images from disk
            for filepath in batch_files:
                img = read_image(filepath, mode=ImageReadMode.RGB).float() / 255.0
                img_list.append(img)
                
            # Grab dims from the first image if not already set
            if not hasattr(self, 'init_dims'):
                self.init_dims = (img_list[0].shape[1], img_list[0].shape[2])
                
            # Stack into a single tensor and push to GPU
            batch_tensor = torch.stack(img_list).to(self.device)
            
            # Apply the augmentations simultaneously to batch
            with torch.no_grad():
                processed_batch = self.aug(batch_tensor)
            
            # Unpack batch and save each image back to disk
            for j, filepath in enumerate(batch_files):
                filename = os.path.basename(filepath)
                dst_path = os.path.join(self.dest_root, filename)
            
                save_image(processed_batch[j], dst_path)


    def get_dataset(self, **dataset_kwargs):
        """
        Generates data if needed, then returns the FlatFolderDataset.
        
        Args:
            **dataset_kwargs: Arguments meant for FlatDataset (e.g. transform)
        """
        # Ensure data exists
        self.prepare()
        
        # Define default transform if not provided
        if 'transform' not in dataset_kwargs or dataset_kwargs['transform'] is None:
            dataset_kwargs['transform'] = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
            
        return FlatDataset(root=self.dest_root, **dataset_kwargs)

def create_tps_transform(strength=0.5, grid_size=5, device='cuda'):
    """
    Precomputes the TPS grid once and returns a batched augmentation function
    that randomly bulges outwards or pinches inwards from a randomised centre.
    """
    coords = torch.linspace(-1, 1, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    base_src_points = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2)
    
    def tps_transform(image):
        B = image.shape[0]
        src_points = base_src_points.expand(B, -1, -1)
        
        cx = -0.5 + 1 * torch.rand((B, 1), device=device)
        cy = -0.5 + 1 * torch.rand((B, 1), device=device)
        radius = 0.4 + 0.1 * torch.rand((B, 1), device=device)
        
        direction = torch.where(torch.rand((B, 1), device=device) > 0.5, 1.0, -1.0)
        
        # Calculate distances from the centre
        dist_x = src_points[..., 0] - cx
        dist_y = src_points[..., 1] - cy
        dist = torch.sqrt(dist_x**2 + dist_y**2)
        
        
        # Calculate the direction to push each pin
        safe_dist = dist + 1e-8 # Prvent division by zero
        dir_x = dist_x / safe_dist
        dir_y = dist_y / safe_dist
        
        # Calculate the bump curve (smooth falloff from centre)
        normalised_dist = dist / radius
        bump_curve = torch.pow(1.0 - torch.pow(normalised_dist, 2), 2)
        
        displacement_amount = strength * bump_curve * direction * radius * 0.5
        
        dst_points = src_points.clone()
        mask = dist < radius
        
        # Push or pull the pins for position
        dst_points[..., 0] += torch.where(mask, displacement_amount * dir_x, torch.tensor(0.0, device=device))
        dst_points[..., 1] += torch.where(mask, displacement_amount * dir_y, torch.tensor(0.0, device=device))
        
        kernel, affine = K_geometry.get_tps_transform(dst_points, src_points)
        return K_geometry.warp_image_tps(image, src_points, kernel, affine)

    return tps_transform
class RandomRicianNoise(nn.Module):
    """
    Applies Rician noise to a PyTorch tensor.
    """
    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Generate independent Gaussian noise for real and imaginary components
        noise_real = torch.randn_like(img) * self.std
        noise_imag = torch.randn_like(img) * self.std
        
        # Calculate the Rician noise magnitude
        noisy_img = torch.sqrt((img + noise_real)**2 + noise_imag**2)
        
        return noisy_img