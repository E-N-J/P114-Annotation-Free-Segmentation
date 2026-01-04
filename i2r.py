import torch
from rpca import RobustPCA
from rda import RobustDeepAutoencoder
from eval import evaluate_models
from helpers.filteredDataset import FilteredDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 50
IMG_SIZE = 168
FOOTAGE_ID = "hall1" 

print(f"Using device: {DEVICE}\n")

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
    
    i2r_data = FilteredDataset(
        root="./datasets/I2R_Frames",
        transform=transform,
        target_class=FOOTAGE_ID
    )
    i2r_loader = DataLoader(
        i2r_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        i2r_data,
        batch_size=len(i2r_data),
        shuffle=False
    )

    rpca = RobustPCA(max_iter=2000, lambda_=None, device=DEVICE) 
    ae = RobustDeepAutoencoder(latent_dim=3, dropout=0, std=0).to(DEVICE)
    
    ae.fit_admm(
        i2r_loader,
        lr=2e-4,
        lambda_=0.1,
        outer_epochs=2,
        inner_epochs=10,
        # norm_type='l21'
    )
    ae.plot_training_curve(log_scale=True)
    ae.eval()
    evaluate_models(test_loader, rpca, ae, device=DEVICE, subject_id=FOOTAGE_ID, results_root="./results/i2r")    
    
    # print("Do you want to save the trained Autoencoder model? (y/n): ", end="")
    # save_choice = input().strip().lower()
    # if save_choice == 'y':
    #     model_name = "rae_model_2"
    #     torch.save(ae.state_dict(), f"{model_name}.pth")
    #     print(f"Model saved as {model_name}.pth")
    # else:
    #     print("Model not saved.")