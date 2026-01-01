import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt


class BaseLoader:
    """
    Base class for loading image datasets from a local file path.
    Supports flexible preprocessing and dataset creation logic.
    """
    def __init__(self, root_path, img_size=128, batch_size=32):
        """
        Args:
            root_path (str): Absolute or relative path to the dataset folder.
            img_size (int): Target size for resizing images.
            batch_size (int): Batch size for loaders.
        """
        self.root_path = root_path
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.full_dataset = None
        
    def get_transforms(self):
        """
        Standard grayscale transforms. Override for color/augmentation.
        """
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def preprocess_data(self, path):
        """
        HOOK: Override this in child classes to modify files 
        BEFORE dataset creation.
        
        """
        pass 

    def create_dataset(self, path):
        """
        FACTORY: Override this to return a custom Dataset object if 
        ImageFolder is not suitable.
        """
        print(f"Attempting to load ImageFolder from '{path}'")
        transform = self.get_transforms()
        return ImageFolder(root=path, transform=transform)

    def setup(self):
        """
        Orchestrates the pipeline: Verify Path -> Preprocess -> Load.
        """
        print(f"1. Verifying path '{self.root_path}'")
        
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"The path '{self.root_path}' does not exist.")
  
        print("2. Checking for preprocessing steps...")
        # Note: preprocess_data might return a new path (e.g., if it creates a 'processed' subfolder)
        # We allow the hook to return a path, or default to the original root_path
        new_path = self.preprocess_data(self.root_path)
        final_path = new_path if new_path else self.root_path
        
        print(f"3. Creating Dataset object from '{final_path}'")
        try:
            self.full_dataset = self.create_dataset(final_path)
            print(f"   Success! Loaded {len(self.full_dataset)} images.")
        except Exception as e:
            print(f"   Error loading dataset: {e}")
            print("   (Hint: If your data isn't in 'class' folders, override `create_dataset`.)")

    def visualize(self, start_idx=0, num_images=16, columns=4, cmap='gray'):
        if self.full_dataset is None:
            print("Error: Dataset not loaded. Please run .setup() first.")
            return

        rows = (num_images // columns) + (1 if num_images % columns != 0 else 0)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3.5))
        axes = axes.flatten()

        print(f"Visualizing images from index {start_idx}...")

        for i in range(num_images):
            dataset_idx = start_idx + i
            if dataset_idx >= len(self.full_dataset): break
            
            data_item = self.full_dataset[dataset_idx]
            img_tensor, label_idx = data_item[0], data_item[1]
            
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            if hasattr(self.full_dataset, 'classes'):
                label_name = self.full_dataset.classes[label_idx]
            else:
                label_name = str(label_idx)

            ax = axes[i]
            ax.imshow(img_np.squeeze(), cmap=cmap)
            ax.set_title(f"Idx: {dataset_idx}\n{label_name}", fontsize=9)
            ax.axis('off')

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    def get_loaders(self):
        raise NotImplementedError("Implement `get_loaders` in child class.")