import os
import cv2
import numpy as np

def save_rpca_results(x, L, S, subject_id, results_root="./results"):
    base_dir = os.path.join(results_root, f"{subject_id}")
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