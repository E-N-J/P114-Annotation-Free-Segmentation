import torch
from rpca import RobustPCA
from rda import RobustDeepAutoencoder
from eval import evaluate_models
from helpers.filteredDataset import FilteredDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 65
IMG_SIZE = 168

SUBJECT_ID = 7
# started with high latent and shadows were in X, reduced to 16, then 8
# kept halving until I reached a limit of reconstruction in X (shadows forced into S) which was at 2 dims
# 3 dims still captured shadows but 2 was too little capacity to reconstruct X properly
# Used dropout to achieve effective 2.5 latent dim
print(f"Using device: {DEVICE}\n")

# TODO 
# fix forced square img dims
# neaten eval function, split it up
# remove noise and spectral norm from RDA


if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
    
    yb_data = FilteredDataset(
        root="./datasets/yaleB",
        transform=transform,
        target_class=SUBJECT_ID
    )
    yb_loader = DataLoader(
        yb_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    rpca = RobustPCA(max_iter=2000, lambda_=None, device=DEVICE) 
    
    #0.1,5,1e-4
    # 1e-3,0.005,20,25 latent 16 on par with RPCA (kinda)
    # test_loader.dataset,
    #     lr=4e-4,
    #     lambda_=0.001,
    #     batch_size=BATCH_SIZE,
    #     outer_epochs=7,
    #     inner_epochs=50, GOOD
    ae = RobustDeepAutoencoder(latent_dim=2, dropout=0, std=0).to(DEVICE)
    ae.fit_admm(
        yb_loader,
        lr=4e-4,
        lambda_=0.9,
        outer_epochs=4,
        inner_epochs=70,
        # norm_type='l21'
    )
    ae.plot_training_curve(log_scale=True)
    ae.eval()
    evaluate_models(yb_loader, rpca, ae, device=DEVICE, subject_id=SUBJECT_ID, results_root="./results/yaleB")    
    
    # print("Do you want to save the trained Autoencoder model? (y/n): ", end="")
    # save_choice = input().strip().lower()
    # if save_choice == 'y':
    #     model_name = "rae_model_2"
    #     torch.save(ae.state_dict(), f"{model_name}.pth")
    #     print(f"Model saved as {model_name}.pth")
    # else:
    #     print("Model not saved.")