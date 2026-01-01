import torch
from loaders.yaleB import YaleBLoader
from rpca import RobustPCA
from rae import RobustDeepAutoencoder
from eval import evaluate_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 65
IMG_SIZE = 168
# LAMBDA_PARAM = 0.0001
SUBJECT_ID = 7 

print(f"Using device: {DEVICE}\n")

if __name__ == "__main__":
    print("Initializing Data Loader...")
    yb = YaleBLoader(
        root_path="./datasets/yaleB",
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    train_loader, test_loader = yb.get_loaders(test_subject_id=SUBJECT_ID)
    yb.visualize(start_idx=0, num_images=6, columns=3)

    rpca = RobustPCA(max_iter=2000, lambda_=None, device=DEVICE) 
    
    ae = RobustDeepAutoencoder().to(DEVICE)
    ae.simple_fit(test_loader, epochs=350, lr=1e-4, patience=10, tol=5, lambda_p=0.1)
    ae.plot_training_curve()


    evaluate_models(test_loader, rpca, ae, device=DEVICE, subject_id=SUBJECT_ID, results_root="./results")
    
    # print("Do you want to save the trained Autoencoder model? (y/n): ", end="")
    # save_choice = input().strip().lower()
    # if save_choice == 'y':
    #     model_name = "rae_model_2"
    #     torch.save(ae.state_dict(), f"{model_name}.pth")
    #     print(f"Model saved as {model_name}.pth")
    # else:
    #     print("Model not saved.")