import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import cv2
import numpy as np

def evaluate_models(dataloader, rpca_model, ae_model, subject_id, device=torch.device("cpu"), results_root="./results"):
    ae_model.eval()
    
    batch = next(iter(dataloader))
    x_cpu, _ = batch 
    batch_size = x_cpu.shape[0]
    print(f"\nEvaluating on batch of size {batch_size}...")

    subject_dir = os.path.join(results_root, f"Subject_{subject_id}")
    lowrank_dir = os.path.join(subject_dir, "LowRank")
    sparse_dir = os.path.join(subject_dir, "Sparse")
   
    L_rpca = None
    S_rpca = None

    if os.path.exists(lowrank_dir) and len(os.listdir(lowrank_dir)) == batch_size:
        print(f"Found pre-computed RPCA results. Loading from disk...")
        
        L_list = []
        S_list = []
        filenames = sorted(os.listdir(lowrank_dir))
        
        for fname in filenames:
            l_img = cv2.imread(os.path.join(lowrank_dir, fname), cv2.IMREAD_GRAYSCALE)
            s_img = cv2.imread(os.path.join(sparse_dir, fname), cv2.IMREAD_GRAYSCALE)
            
            # Normalise to [0,1]
            
            L_list.append(torch.tensor(l_img / 255.0, dtype=torch.float32))
            S_list.append(torch.tensor(s_img / 255.0, dtype=torch.float32))
            
        # Stack into (N, 1, H, W)
        L_rpca = torch.stack(L_list).unsqueeze(1)
        S_rpca = torch.stack(S_list).unsqueeze(1)
        
    else:
        print("No saved results found. Running RPCA Inference...")
        
        x_gpu = x_cpu.to(device)
        
        L_gpu, S_gpu = rpca_model.decompose(x_gpu)
        
        L_rpca = L_gpu.detach().cpu()
        S_rpca = S_gpu.detach().cpu()
        
        save_rpca_results(x_cpu, L_rpca, S_rpca, subject_id, results_root)

    print("Running Autoencoder Inference...")
    with torch.no_grad():
        x_gpu = x_cpu.to(device)
        L_ae_gpu = ae_model(x_gpu)
        
        L_ae = L_ae_gpu.detach().cpu()
        
        S_ae = x_cpu - L_ae 
    print(f"Autoencoder Inference done.\n")

    fig, ax = plt.subplots(3, 3, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.2) 
    state = {'idx': 0}

    def show_img(ax_idx, img, title):
        ax_idx.clear()
        ax_idx.imshow(img.squeeze(), cmap='gray')
        ax_idx.set_title(title)
        ax_idx.axis('off')

    def update_plot(idx):
        # Original
        show_img(ax[0,0], x_cpu[idx], f"Input [{idx}]")
        ax[0,1].axis('off'); ax[0,2].axis('off')

        # RPCA
        show_img(ax[1,0], L_rpca[idx], "RPCA L (Background)")
        show_img(ax[1,1], abs(S_rpca[idx]), "RPCA S (Anomalies)")
        ax[1,2].axis('off')

        # Autoencoder
        show_img(ax[2,0], L_ae[idx], "Autoencoder L")
        show_img(ax[2,1], abs(S_ae[idx]), "AE S (Residual)")
        ax[2,2].axis('off')
        
        fig.canvas.draw_idle()

    # Buttons
    ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
    ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
    b_prev = Button(ax_prev, 'Previous')
    b_next = Button(ax_next, 'Next')
    
    def next_click(event):
        state['idx'] = (state['idx'] + 1) % batch_size
        update_plot(state['idx'])
    def prev_click(event):
        state['idx'] = (state['idx'] - 1) % batch_size
        update_plot(state['idx'])
        
    b_next.on_clicked(next_click)
    b_prev.on_clicked(prev_click)

    update_plot(0)
    plt.show()

def save_rpca_results(x, L, S, subject_id, results_root="./results"):
    base_dir = os.path.join(results_root, f"Subject_{subject_id}")
    dirs = {
        "Original": os.path.join(base_dir, "Original"),
        "LowRank": os.path.join(base_dir, "LowRank"),
        "Sparse": os.path.join(base_dir, "Sparse")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
        
    print(f"Saving RPCA results to {base_dir}...")

    def process_and_save(tensor, folder, idx):
        img = tensor[idx].squeeze().detach().cpu().numpy()
        
        if "Sparse" in folder: img = np.abs(img)
            
        img_min, img_max = img.min(), img.max()
        if img_max > img_min: 
            img = (img - img_min) / (img_max - img_min)
        
        img = (img * 255).astype(np.uint8)
        fname = f"{idx:03d}.png"
        cv2.imwrite(os.path.join(folder, fname), img)

    batch_size = x.shape[0]
    for i in range(batch_size):
        process_and_save(x, dirs["Original"], i)
        process_and_save(L, dirs["LowRank"], i)
        process_and_save(S, dirs["Sparse"], i)
        
    print("Save Complete.")