import os
import cv2
import numpy as np

def save_rpca_results(x, L, S, results_root="./results"):
    dirs = {
        "Original": os.path.join(results_root, "Original"),
        "LowRank": os.path.join(results_root, "LowRank"),
        "Sparse": os.path.join(results_root, "Sparse")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
        
    print(f"Saving RPCA results to {results_root}...")

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