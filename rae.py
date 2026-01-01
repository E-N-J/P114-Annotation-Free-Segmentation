import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

class RobustDeepAutoencoder(nn.Module):
    """
    Robust Deep Autoencoder for decomposing images into low-rank and sparse components.
    Inspired by Zhou & Paffenroth, "Anomaly Detection with Robust Deep Autoencoders.
    """
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 168x168 -> 84x84
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 84x84 -> 42x42
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 42x42 -> 21x21
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.flatten_dim = 64 * 21 * 21
        self.latent_dim = 1024
        
        self.fc_encoder = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_decoder = nn.Linear(self.latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            # 21x21 -> 42x42
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 42x42 -> 84x84
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 84x84 -> 168x168
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid() 
        )
        
        self.total_loss_history = []
        self.l2_loss_history = []
        self.l1_loss_history = []
        
    def robust_loss(self, x, L_pred, lambda_p=0.1):
        """
        Loss = ||S|| + lambda * ||S||_1
        Assuming S = x - L (residual), we want to minimize L2 of reconstruction
        BUT allow for large sparse errors (L1 penalty on residual).
        """
        residual = x - L_pred
        l2_loss = torch.norm(residual, p=2)
        l1_loss = torch.norm(residual, p=1)
        
        loss = l2_loss + lambda_p * l1_loss
        return loss, l2_loss, l1_loss

    def forward(self, x):
        x = self.encoder(x)
        
        x_flat = torch.flatten(x, start_dim=1) 
        latent = self.fc_encoder(x_flat)
        
        x_expand = self.fc_decoder(latent)
        x_reshaped = x_expand.view(-1, 64, 21, 21)
        
        L_pred = self.decoder(x_reshaped)
        return L_pred
    
    def simple_fit(self, loader, epochs=10, lr=1e-3, patience=5, tol=5, lambda_p=0.1):
        """
        Autoencoder training with early stopping based on a simple robust loss (not specified in paper).
        """
        print(f"\nTraining Robust Autoencoder (Patience={patience})...")
        optimizer = optim.Adam(self.parameters(), lr=lr)
        device = next(self.parameters()).device
        
        self.total_loss_history = []
        self.l2_loss_history = []
        self.l1_loss_history = []
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        for epoch in range(epochs): 
            total_loss = 0
            total_l2 = 0
            total_l1 = 0
            progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}", leave=True)
            
            for _, (x, _) in progress_bar:
                x = x.to(device)
                optimizer.zero_grad()
                
                L_pred = self(x)
                
                loss = l2_loss + (lambda_p * l1_loss)
                
                loss, l2_loss, l1_loss = self.robust_loss(x, L_pred, lambda_p=lambda_p)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_l2 += l2_loss.item()
                total_l1 += l1_loss.item()
                
                best_str = f"{best_loss:.4f}" if best_loss != float('inf') else "inf"
                progress_bar.set_postfix({
                    'total_loss': f'{loss.item():.4f}',
                    'l2_loss': f'{l2_loss.item():.4f}',
                    'l1_loss': f'{l1_loss.item():.4f}',
                    'best': best_str,
                    'pat': f'{patience_counter}/{patience}'
                })
                
            avg_loss = total_loss / len(loader)
            avg_l2 = total_l2 / len(loader)
            avg_l1 = total_l1 / len(loader)
            self.total_loss_history.append(avg_loss)
            self.l2_loss_history.append(avg_l2)
            self.l1_loss_history.append(avg_l1)
            
            if avg_loss < best_loss - tol:
                best_loss = avg_loss
                best_avgs = (avg_loss, avg_l2, avg_l1)
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:

                    print("\nEarly stopping triggered!")
                    print(f"Restoring best model from Epoch {epoch+1 - patience} (Losses Total, (Total, L2, L1)): {best_avgs}")
                    self.load_state_dict(best_model_weights)
                    break
            
            
        print(f"Training Complete. Final Avg Losses: {best_avgs}")
    
    def plot_training_curve(self):
        """
        Visualizes the training loss over epochs.
        """
        if not self.total_loss_history:
            print("No training history found. Run a fit first.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.total_loss_history, linestyle='-', color='b', label='Training Loss')
        if self.l2_loss_history and self.l1_loss_history:
            plt.plot(self.l2_loss_history, linestyle='--', color='r', label='L2 (Rec) Loss')
            plt.plot(self.l1_loss_history, linestyle='--', color='g', label='L1 (Sparse) Loss')
        plt.title('Robust Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (L2 + lambda*L1)')
        plt.grid(True)
        # plt.yscale('log')
        plt.legend()
        plt.show()
