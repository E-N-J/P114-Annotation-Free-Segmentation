import torch
import torch.nn.functional as F
import os
from .utils import save_rpca_results
from contextlib import contextmanager

def get_rpca_decomposition(X_input, rpca_model, results_root, force_recompute=False, target_size=None):
    """Handles the RPCA decomposition, resizing on the GPU where possible."""
    
    try:
        device = next(rpca_model.parameters()).device
    except StopIteration:
        device = next(rpca_model.buffers()).device

    use_cached = False

    if os.path.exists(results_root) and os.path.exists(os.path.join(results_root, "rpca_results.pt")) and not force_recompute:
        print(f"Found pre-computed RPCA results. Loading from disk...")
        rpca_results = torch.load(os.path.join(results_root, "rpca_results.pt"))
        if rpca_results['L'].shape[0] == X_input.shape[0]:
            use_cached = True
        else:
            print("Warning: RPCA results batch size does not match input batch size. Recomputing RPCA...")
            use_cached = False
    else:
        print("No saved results found. Computing RPCA and saving to disk...")
        use_cached = False
        
    if use_cached:
        L_rpca = rpca_results['L']
        S_rpca = rpca_results['S']
        X_cache = rpca_results['X'] if 'X' in rpca_results else None
        X_input = X_cache if X_cache is not None else F.interpolate(X_input, size=target_size).cpu() #TODO: can crash if no target size is provided
    else:
        print("Running RPCA Inference...")
        X_gpu = X_input.to(device)
        L_gpu, S_gpu = rpca_model.decompose(X_gpu, fast=True, cols=False)

        if target_size is not None:
            L_rpca = F.interpolate(L_gpu, size=target_size).cpu()
            S_rpca = F.interpolate(S_gpu, size=target_size).cpu()
            X_input = F.interpolate(X_gpu, size=target_size).cpu()
        else:
            X_input, L_rpca, S_rpca = X_input.cpu(), L_gpu.detach().cpu(), S_gpu.detach().cpu()

        save_rpca_results(X_input, L_rpca, S_rpca, results_root)
            
    return X_input, L_rpca, S_rpca

@contextmanager
def inference_generator(model):
    """A lightweight context manager to standardise the API for non-ceVAE models."""
    def process_batch(batch_x):
        with torch.no_grad():
            output = model(batch_x)
            L_dev = output[0] if isinstance(output, tuple) else output
            S_dev = batch_x - L_dev
        return L_dev.detach(), S_dev.detach()
    
    yield process_batch

def run_deep_models_inference(X_input, models_dict, target_size=None, batch_size=128):
    """Runs inference for deep learning models using batching."""
    print("Running Deep Models Inference...")
    model_results = {}
    n_samples = X_input.size(0)
    
    for name, model in models_dict.items():
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = next(model.buffers()).device
        
        L_batches = []
        S_batches = []
        if name == 'ceVAE':
            print(f"Running {name} inference with guided backprop and noise tunnel...")
            selected_inference = model.anomaly_generator(
                nt_samples_batch_size=10,
                nt_samples=50,
            )
        else:
            print(f"Running {name} inference with standard approach in batches...")
            selected_inference = inference_generator(model)

        with selected_inference as infer:
            
            for i in range(0, n_samples, batch_size):
                batch_x = X_input[i:i + batch_size].to(device, non_blocking=True)
                
                L_device, S_device = infer(batch_x)
                
                if target_size is not None:
                    L_device = F.interpolate(L_device, size=target_size)
                    S_device = F.interpolate(S_device, size=target_size)

                L_batches.append(L_device.cpu())
                S_batches.append(S_device.cpu())
            
        model_results[name] = [torch.cat(L_batches, dim=0), torch.cat(S_batches, dim=0)]
            
    print("Inference done.\n")
    return model_results

def calculate_dice(truth, S, threshold=0.05):
    """
    Calculates the Dice Similarity Coefficient between ground truth and predicted sparse anomalies.
    Assumes both inputs are already on the CPU.
    """
    y_true = torch.as_tensor(truth).flatten()
    y_scores = torch.as_tensor(S).flatten()
    
    # Binarise the ground truth
    y_true = (y_true > 0.5).float()
    
    # Binarise the model predictions
    y_pred = (y_scores.abs() > threshold).float()

    intersection = torch.sum(y_true * y_pred)
    denominator = torch.sum(y_true) + torch.sum(y_pred)
    
    epsilon = 1e-8
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    
    return dice.item()

def find_optimal_dice(truth, S, bins=1000):
    """
    Finds the optimal Dice threshold using highly efficient histogram binning.
    Evaluates 1000 threshold steps in milliseconds.
    """
    
    y_true = torch.as_tensor(truth).flatten().bool()
    y_scores = torch.as_tensor(S).flatten().abs()
    
    min_val = y_scores.min().item()
    max_val = y_scores.max().item()
    
    scores_anomaly = y_scores[y_true]
    scores_normal = y_scores[~y_true]
    
    total_true = scores_anomaly.numel()
    
    if total_true == 0:
        return 0.0, 0.0 # No anomalies, return default threshold and zero Dice
    
    hist_anomaly = torch.histc(scores_anomaly, bins=bins, min=min_val, max=max_val)
    hist_normal = torch.histc(scores_normal, bins=bins, min=min_val, max=max_val)
    
    tp = torch.flip(torch.cumsum(torch.flip(hist_anomaly, dims=[0]), dim=0), dims=[0])
    fp = torch.flip(torch.cumsum(torch.flip(hist_normal, dims=[0]), dim=0), dims=[0])
    
    pred_positives = tp + fp
    epsilon = 1e-8
    dice_scores = (2.0 * tp + epsilon) / (total_true + pred_positives + epsilon)
    
    best_idx = torch.argmax(dice_scores)
    best_dice = dice_scores[best_idx].item()
    
    thresholds = torch.linspace(min_val, max_val, steps=bins)
    best_threshold = thresholds[best_idx].item()
    
    return best_threshold, best_dice