import torch
import os
import cv2
from .vis import visualise_results
from .utils import save_rpca_results

def evaluate_models(dataloader, rpca_model, models_dict, subject_id, results_root="./results"):
    for model in models_dict.values():
        model.eval()
  
    batch = next(iter(dataloader))
    X_input, _, _ = batch 
    print(f"\nEvaluating on batch of size {X_input.shape[0]}...")
    results_root = os.path.abspath(results_root)
   
    L_rpca, S_rpca = _get_rpca_decomposition(X_input, rpca_model, subject_id, results_root)
    
    model_results = _run_deep_models_inference(X_input, models_dict)
    
    visualise_results(X_input, L_rpca, S_rpca, model_results)
    
def _get_rpca_decomposition(X_input, rpca_model, subject_id, results_root):
    """Handles the RPCA decomposition, either by loading from disk or by running inference."""
    
    subject_dir = os.path.join(results_root, f"{subject_id}")
    lowrank_dir = os.path.join(subject_dir, "LowRank")
    sparse_dir = os.path.join(subject_dir, "Sparse")
    device = next(rpca_model.buffers()).device

    batch_size = X_input.shape[0]
    if os.path.exists(lowrank_dir) and len(os.listdir(lowrank_dir)) == batch_size:
        print(f"Found pre-computed RPCA results. Loading from disk...")
        
        L_list, S_list = [], []
        for fname in sorted(os.listdir(lowrank_dir)):
            l_img = cv2.imread(os.path.join(lowrank_dir, fname), cv2.IMREAD_GRAYSCALE)
            s_img = cv2.imread(os.path.join(sparse_dir, fname), cv2.IMREAD_GRAYSCALE)
            
            L_list.append(torch.tensor(l_img / 255.0, dtype=torch.float32))
            S_list.append(torch.tensor(s_img / 255.0, dtype=torch.float32))
            
        L_rpca = torch.stack(L_list).unsqueeze(1)
        S_rpca = torch.stack(S_list).unsqueeze(1)
    else:
        print("No saved results found. Running RPCA Inference...")
        X_gpu = X_input.to(device)
        L_gpu, S_gpu = rpca_model.decompose(X_gpu, fast=True, cols=False)
        L_rpca, S_rpca = L_gpu.detach().cpu(), S_gpu.detach().cpu()
        save_rpca_results(X_input, L_rpca, S_rpca, subject_id, results_root)
        
    return L_rpca, S_rpca

def _run_deep_models_inference(X_input, models_dict):
    """Runs inference for the deep learning models."""
    print("Running Deep Models Inference...")
    model_results = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = next(model.buffers()).device
            
            X_gpu = X_input.to(device)
            output = model(X_gpu)
            if isinstance(output, tuple):
                output = output[0]
            L_gpu = output
            L_cpu = L_gpu.detach().cpu()
            S_cpu = X_input - L_cpu
            model_results[name] = (L_cpu, S_cpu)
    print(f"Inference done.\n")
    return model_results