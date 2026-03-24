import os
import torch

def save_rpca_results(x, L, S, results_root="./results"):
    os.makedirs(results_root, exist_ok=True)
    save_path = os.path.join(results_root, "rpca_results.pt")
    torch.save({'L': L.cpu(), 'S': S.cpu(), 'X': x.cpu()}, save_path)
    print(f"RPCA results saved to {save_path}.")