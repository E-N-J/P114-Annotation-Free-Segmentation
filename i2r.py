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
    ae_params = {
        'latent_dim': 8,
        'dropout': 0.33,
        'std': 0.5,
        'lr': 2e-4,
        'lambda_': 0.003,
        'outer_epochs': 12,
        'inner_epochs': 70,
    }
    rpca = RobustPCA(max_iter=6000, lambda_=None, device=DEVICE, tol=1e-7) 
    ae = RobustDeepAutoencoder(latent_dim=ae_params['latent_dim'], dropout=ae_params['dropout'], std=ae_params['std']).to(DEVICE)
    print(f"Using parameters: {ae_params}\n")
    
    ae.fit_admm(
        i2r_loader,
        lr=ae_params['lr'],
        lambda_=ae_params['lambda_'],
        outer_epochs=ae_params['outer_epochs'],
        inner_epochs=ae_params['inner_epochs'],
        # norm_type='l21'
    )
    ae.plot_training_curve(display=False, log_scale=True)
    ae.eval()
    # evaluate_models(i2r_loader, rpca, ae, device=DEVICE, subject_id=FOOTAGE_ID, results_root="./results/i2r")
    evaluate_models(test_loader, rpca, ae, device=DEVICE, subject_id=FOOTAGE_ID, results_root="./results/i2r")
    
    print("Do you want to save the trained Autoencoder model? (y/n): ", end="")
    save_choice = input().strip().lower()
    if save_choice == 'y':
        model_name = "rda_model_i2r_3"
        torch.save(ae.state_dict(), f"{model_name}.pth")
        print(f"Model saved as {model_name}.pth")
    else:
        print("Model not saved.")