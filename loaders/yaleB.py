from torch.utils.data import DataLoader, Subset
from .base import BaseLoader

class YaleBLoader(BaseLoader):
    """
    Specific loader for the Extended YaleB dataset.
    """

    def get_loaders(self, test_subject_id=0):

        if self.full_dataset is None:
            self.setup()
            
        train_indices = [
            i for i, (path, label) in enumerate(self.full_dataset.samples) 
            if label != test_subject_id
        ]
        
        train_set = Subset(self.full_dataset, train_indices)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)

        test_indices = [
            i for i, (path, label) in enumerate(self.full_dataset.samples)
            if label == test_subject_id
        ]

        test_set = Subset(self.full_dataset, test_indices)
        
        # Batch size = Full Dataset (needed for RPCA)
        bs = len(test_set) if len(test_set) > 0 else 1
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

        return train_loader, test_loader
    
# Example usage:
# if __name__ == "__main__":
#     ybl = YaleBLoader(root_path="./datasets/yaleB", img_size=168, batch_size=8)
#     ybl.setup()
#     ybl.visualize(start_idx=0, num_images=5)