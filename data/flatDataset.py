import os
import glob
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class FlatDataset(Dataset):
    """
    1. Loads images from a flat directory (no class subfolders required).
    2. Returns (image, label, LOCAL_INDEX) for ADMM compatibility.
    """
    def __init__(self, root, transform=None, target_class=None):
        """
        Args:
            root (str): Path to the folder containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_class: Ignored for FlatFolderDataset (kept for compatibility).
        """
        self.root = root
        self.transform = transform

        self.samples = sorted(glob.glob(os.path.join(root, "*.*")))
        self.samples = [f for f in self.samples if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 files in {root}. Check your path.")
            
        # Mock 'classes' attribute for the visualize method
        # Since it's a flat folder, we assume everything belongs to one class.
        self.classes = [os.path.basename(root)]
        self.targets = [0] * len(self.samples) # All images have label 0

        print(f"Loaded Flat Dataset: {len(self.samples)} images from '{self.classes[0]}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is always 0.
        """
        path = self.samples[index]
        
        #Load Image Grayscale 'L', color 'RGB'
        img = Image.open(path).convert('L')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # Return (Image, Label, Index)
        # '0' as the label because we only have one class
        return img, 0, index

    def visualize(self, start_idx=0, num_images=16, columns=4, cmap='gray'):
        """
        Visualizes a grid of images from the current dataset.
        Copied and adapted from FilteredDataset.
        """
        if len(self) == 0:
            print("Dataset is empty.")
            return

        rows = (num_images // columns) + (1 if num_images % columns != 0 else 0)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3.5))
        
        if num_images == 1: axes = [axes]
        elif isinstance(axes, (list, object)) and hasattr(axes, 'flatten'): axes = axes.flatten()
        elif isinstance(axes, plt.Axes): axes = [axes] # Handle single subplot case

        print(f"Visualizing images from local index {start_idx}...")

        for i in range(num_images):
            dataset_idx = start_idx + i
            if dataset_idx >= len(self): break

            # Unpack the tuple (img, label, idx)
            img, label, idx = self[dataset_idx]

            # Convert Tensor to Numpy for Plotting
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img)

            label_name = self.classes[label]

            ax = axes[i]
            ax.imshow(img_np.squeeze(), cmap=cmap)
            ax.set_title(f"Idx: {idx}\n{label_name}", fontsize=9)
            ax.axis('off')

        # Turn off remaining empty axes
        if hasattr(axes, '__len__'):
            for j in range(i + 1, len(axes)): 
                axes[j].axis('off')
                
        plt.tight_layout()
        plt.show()