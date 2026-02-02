import torchvision.transforms as transforms
import os
import glob
import torch
import kornia.augmentation as K_transforms
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from .flatDataset import FlatDataset 

class Augmentor:
    def __init__(self, source_root, rotation=0, tps=0, elastic=0, device='cuda'):
        """
        Args:
            source_root: Path to the flat folder of raw images.
            rotation: Degrees of random rotation.
            tps: Scale of Thin Plate Spline distortion.
            elastic: Scale of Elastic Transform distortion.
            device: Device to run augmentations on.
            """
        self.source_root = source_root.rstrip(os.sep)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.is_identity = (rotation == 0 and tps == 0 and elastic == 0)

        if not self.is_identity:
            # Define the GPU Augmentation Pipeline
            augs = []
            if rotation > 0:
                augs.append(K_transforms.RandomRotation(degrees=rotation, p=1.0))
            if tps > 0:
                augs.append(K_transforms.RandomThinPlateSpline(scale=tps, align_corners=False, p=1.0))
            if elastic > 0:
                augs.append(K_transforms.RandomElasticTransform(alpha=elastic * 50.0, p=1.0))
                
            self.aug = torch.nn.Sequential(*augs).to(self.device)

            # Construct the destination folder name
            folder_name = os.path.basename(self.source_root)
            suffix = f"_Rot{rotation}_TPS{tps}_El{elastic}"
            parent_dir = os.path.dirname(self.source_root)
            self.dest_root = os.path.join(parent_dir, folder_name + suffix)
        else:
            # If no augmentation, the destination is just the source
            self.dest_root = self.source_root

    def prepare(self, force_rebuild=False):
        """
        Runs the augmentation pipeline and saves images to disk.
        Skips if the folder already exists.
        """
        # Skip if Identity or already exists
        if self.is_identity:
            print(f"Identity augmentation requested. Using raw source: {self.source_root}")
            return

        if os.path.exists(self.dest_root) and not force_rebuild:
            print(f"Augmented dataset found at: {self.dest_root}")
            return

        print(f"Generating augmented data to: {self.dest_root}")
        os.makedirs(self.dest_root, exist_ok=True)
        
        # Find files in the flat source folder
        files = glob.glob(os.path.join(self.source_root, "*.*"))
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not valid_files:
            raise ValueError(f"No images found in {self.source_root}. Ensure it is a flat folder.")

        # Augment Loop
        for filepath in tqdm(valid_files, desc="Augmenting"):
            filename = os.path.basename(filepath)
            dst_path = os.path.join(self.dest_root, filename)
            
            img = read_image(filepath, mode=ImageReadMode.RGB).float() / 255.0
            self.init_dims = (img.shape[1], img.shape[2]) # H, W
            img = img.unsqueeze(0).to(self.device) # Shape: (1, 3, H, W)
            
            with torch.no_grad():
                processed = self.aug(img)
            
            # Save
            save_image(processed.squeeze(0), dst_path)

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