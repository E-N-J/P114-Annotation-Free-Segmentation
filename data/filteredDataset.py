import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

class FilteredDataset(ImageFolder):
    """
    1. Loads a standard ImageFolder.
    2. Filters it to keep ONLY a specific class (by Name or Index).
    3. Returns (image, label, LOCAL_INDEX) for ADMM compatibility.
    """
    def __init__(self, root, target_class=None, transform=None):
        super().__init__(root, transform=transform)
        
        if target_class is not None:
            if isinstance(target_class, str):
                if target_class not in self.class_to_idx:
                    available = list(self.class_to_idx.keys())[:5]
                    raise ValueError(f"Class '{target_class}' not found. Did you mean one of {available}...")
                target_idx = self.class_to_idx[target_class]
                
            elif isinstance(target_class, int):
                target_idx = target_class
            else:
                raise TypeError("target_class must be a String (name) or Int (index).")

            self.samples = [
                (path, label) for path, label in self.samples 
                if label == target_idx
            ]
            
            # Update targets list to match
            self.targets = [label for _, label in self.samples]

            print(f"Filtered dataset for Class '{self.classes[target_idx]}' (ID {target_idx}): {len(self.samples)} images.")
        else:
            print(f"Loaded full dataset: {len(self.samples)} images.")

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        
        return image, label, index

    def visualize(self, start_idx=0, num_images=16, columns=4, cmap='gray'):
        """
        Visualizes a grid of images from the current (filtered) dataset.
        """
        # Safety check for empty dataset
        if len(self) == 0:
            print("Dataset is empty.")
            return

        rows = (num_images // columns) + (1 if num_images % columns != 0 else 0)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3.5))
        
        if num_images == 1: axes = [axes]
        elif isinstance(axes, (list, object)): axes = axes.flatten()

        print(f"Visualizing images from local index {start_idx}...")

        for i in range(num_images):
            dataset_idx = start_idx + i
            if dataset_idx >= len(self): break

            img, label, idx = self[dataset_idx]

            img_np = img.permute(1, 2, 0).numpy()
            label_idx = label.item()
            real_idx = idx.item()
            
            label_name = self.classes[label_idx] if self.classes else str(label_idx)

            ax = axes[i]
            ax.imshow(img_np.squeeze(), cmap=cmap)
            ax.set_title(f"Idx: {real_idx}\n{label_name}", fontsize=9)
            ax.axis('off')

        # Turn off remaining empty axes
        if hasattr(axes, '__len__'):
            for j in range(i + 1, len(axes)): 
                axes[j].axis('off')
                
        plt.tight_layout()
        plt.show()