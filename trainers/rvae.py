import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from trainers.base import BaseTrainer


class VAETrainer(BaseTrainer):
    def __init__(self, model, loader):
        super().__init__(model, loader)
        
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

    def fit(self, lr=1e-3, epochs=10):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            loop = tqdm(self.loader, desc=f"VAE Epoch {epoch+1}/{epochs}")
            for batch in loop:
                x = batch[0].to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(x)
                loss, bce, kld = self.loss_function(recon, x, mu, logvar)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                self.global_step += 1
                
                loop.set_postfix({'Loss': loss.item()})
            
            avg_loss = total_loss / len(self.loader.dataset)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            self.histories['loss'] = self.histories.get('loss', []) + [avg_loss]
            
            if self.is_notebook:
                self.plot_metrics(live=True)