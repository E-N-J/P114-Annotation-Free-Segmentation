import os
import glob
import torch
import kornia.augmentation as K_transforms
import kornia.geometry.transform as K_geometry
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from .flatDataset import FlatDataset

# TODO: data aug and saving is slow. consider gpu for concat and parallel disk write
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
        self.device = device

        self.train_folder_name = "input"
        self.gc_folder_name = "groundtruth"
        
        self.is_identity = (rotation == 0 and tps == 0 and elastic == 0)

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
            self.aug = transforms.Compose(augs)

            # Construct the destination folder name
            folder_name = os.path.basename(self.source_root)
            suffix = f"_Rot{rotation}_TPS{tps}_El{elastic}"
            parent_dir = os.path.dirname(self.source_root)
            self.dest_root = os.path.join(parent_dir, folder_name + suffix)
        else:
            # If no augmentation, the destination is just the source
            self.dest_root = self.source_root
            
        files = glob.glob(os.path.join(self.source_root, self.train_folder_name, "*.*"))
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if valid_files:
            tmp_img = read_image(valid_files[0], mode=ImageReadMode.RGB)
            self.init_dims = (tmp_img.shape[1], tmp_img.shape[2]) # H, W
            print(f"Initialized Augmentor with source: {self.source_root}, initial dimensions: {self.init_dims}")
        else:
            raise ValueError(f"No valid images found in {self.source_root}/{self.train_folder_name}. Ensure it is a flat folder.")

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
        os.makedirs(os.path.join(self.dest_root, self.train_folder_name), exist_ok=True)
        os.makedirs(os.path.join(self.dest_root, self.gc_folder_name), exist_ok=True)
        
        train_files = glob.glob(os.path.join(self.source_root, self.train_folder_name , "*.*"))
        valid_train_files = [f for f in train_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        gt_files = glob.glob(os.path.join(self.source_root, self.gc_folder_name , "*.*"))
        valid_gt_files = [f for f in gt_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not valid_train_files:
            raise ValueError(f"No images found in {self.source_root}/{self.train_folder_name} or {self.source_root}/{self.gc_folder_name}. Ensure they are flat folders.")

        if not valid_gt_files:
            print(f"Warning: No images found in {self.source_root}/{self.gc_folder_name}. Ground truth augmentation will be skipped.")

        # Process the files in chunks defined by batch_size
        for i in tqdm(range(0, len(valid_train_files), batch_size), desc="Augmenting in Batches"):
            batch_files = valid_train_files[i : i + batch_size]
            img_list = []
            
            # Read a batch of images from disk
            for filepath in batch_files:
                train_img = read_image(filepath, mode=ImageReadMode.RGB).float() / 255.0
                if valid_gt_files:    
                    gc_img = read_image(filepath.replace(self.train_folder_name, self.gc_folder_name), mode=ImageReadMode.GRAY).float() / 255.0
                    img = torch.cat([train_img, gc_img], dim=0) # C+1, H, W
                    img_list.append(img)
                else:
                    img_list.append(train_img) # C, H, W
                
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
                train_dst_path = os.path.join(self.dest_root, self.train_folder_name, filename)
                if valid_gt_files:
                    gc_dst_path = os.path.join(self.dest_root, self.gc_folder_name, filename)
                    gc_batch = processed_batch[j][3].cpu() # 1, H, W #unsqueeze?
                    train_batch = processed_batch[j][:3].cpu() # C, H, W
                    save_image(train_batch, train_dst_path)
                    save_image(gc_batch, gc_dst_path)
                else:
                    train_batch = processed_batch[j].cpu() # C, H, W
                    save_image(train_batch, train_dst_path)

        print(f"Augmentation complete. Augmented data saved to: {self.dest_root}")


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
            
        return FlatDataset(root=os.path.join(self.dest_root, self.train_folder_name), **dataset_kwargs)

    def get_gc_images(self, num_images=None):
        # Returns the ground truth images as a tensor, if they exist
        gc_folder = os.path.join(self.dest_root, self.gc_folder_name)
        if not os.path.exists(gc_folder):
            print(f"No ground truth folder found at {gc_folder}. Returning None.")
            # return an empty tensor with the correct size to avoid errors in evaluation
            return torch.empty(0)
        gc_files = glob.glob(os.path.join(gc_folder, "*.*"))
        valid_gc_files = [f for f in gc_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not valid_gc_files:
            print(f"No valid ground truth images found in {gc_folder}. Returning None.")
            return torch.empty(0)
        gc_imgs = []
        for filepath in valid_gc_files:
            gc_img = read_image(filepath, mode=ImageReadMode.GRAY).float() / 255.0
            gc_imgs.append(gc_img)
        if num_images is not None:
            gc_imgs = gc_imgs[:num_images]
        return torch.stack(gc_imgs)

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