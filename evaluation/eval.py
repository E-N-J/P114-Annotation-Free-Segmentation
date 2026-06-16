import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import os
from .utils import save_rpca_results
from contextlib import contextmanager

def get_rpca_decomposition(X_input, rpca_model, results_root, force_recompute=False, target_size=None, exact=False):
    """
    Load or compute the RPCA decomposition for a batch of images.

    Args:
        X_input (torch.Tensor): Input batch shaped as (B, C, H, W) on CPU or GPU.
        rpca_model: RPCA model instance used to compute the low-rank and sparse terms.
        results_root (str): Directory used to cache RPCA outputs on disk.
        force_recompute (bool): If True, ignore any cached RPCA result.
        target_size (tuple or None): Optional output size for interpolation. If None,
            the original resolution is preserved.
        exact (bool): If True, use the exact ALM solver instead of iALM.

    Returns:
        tuple: (X_input, L_rpca, S_rpca) as CPU tensors.

    Notes:
        Cached results are reused when the batch size matches. If the cache does not
        contain the original input tensor, the function falls back to the current input.
    """
    
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
        if X_cache is not None:
            X_input = X_cache
        elif target_size is not None:
            X_input = F.interpolate(X_input, size=target_size).cpu()
        else:
            X_input = X_input.cpu()
    else:
        print("Running RPCA Inference...")
        X_gpu = X_input.to(device)
        if exact:
            L_gpu, S_gpu = rpca_model.decompose_ealm(X_gpu)
        else:
            L_gpu, S_gpu = rpca_model.decompose_ialm(X_gpu)

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
    """Yield a batch inference function for models that only return reconstructions."""
    def process_batch(batch_x):
        with torch.no_grad():
            output = model(batch_x)
            L_dev = output[0] if isinstance(output, tuple) else output
            S_dev = torch.abs(batch_x - L_dev)
        return L_dev.detach(), S_dev.detach()
    
    yield process_batch

def run_deep_models_inference(X_input, models_dict, target_size=None, batch_size=128):
    """
    Run batched inference for every model in `models_dict`.

    Args:
        X_input (torch.Tensor): Input batch shaped as (B, C, H, W).
        models_dict (dict): Mapping from model name to model instance.
        target_size (tuple or None): Optional spatial size to interpolate outputs to.
        batch_size (int): Inference batch size.

    Returns:
        dict: Mapping from model name to [L, S] tensors on CPU.
    """
    print("Running Deep Models Inference...")
    model_results = {}
    n_samples = X_input.size(0)
    
    for name, model in models_dict.items():
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = next(model.buffers()).device
            
        torch.cuda.empty_cache()
        
        L_batches = []
        S_batches = []
        if hasattr(model, 'anomaly_generator'): 
            print(f"Running {name} inference with custom anomaly generator...")
            selected_inference = model.anomaly_generator()
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
    Compute Dice overlap between binary ground truth and thresholded anomaly scores.
    """
    y_true = torch.as_tensor(truth).flatten()
    y_scores = torch.as_tensor(S).flatten()
    
    # Binarise the ground truth
    y_true = torch.as_tensor(truth).flatten().bool()
    
    # Binarise the model predictions
    y_pred = (y_scores.abs() > threshold).float()

    intersection = torch.sum(y_true * y_pred)
    denominator = torch.sum(y_true) + torch.sum(y_pred)
    
    epsilon = 1e-8
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    
    return dice.item()

def find_optimal_dice(truth, S, bins=1000):
    """
    Search for the Dice-maximising threshold using histogram binning.

    This is a fast approximation that evaluates `bins` candidate thresholds.
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

def calculate_auroc(truth, S):
    """Compute AUROC from flattened truth labels and anomaly scores."""
    y_true = torch.as_tensor(truth).flatten() # torchmetrics prefers ints for labels
    y_scores = torch.as_tensor(S).flatten().abs()

    metric = BinaryAUROC()
    metric.update(y_scores, y_true)  
    
    # Calculate and return as a standard Python float
    return metric.compute().item()

def calculate_auprc(truth, S):
    """Compute average precision from flattened truth labels and anomaly scores."""
    y_true = torch.as_tensor(truth).flatten() 
    y_scores = torch.as_tensor(S).flatten().abs()

    metric = BinaryAveragePrecision()
    metric.update(y_scores, y_true)  
    
    return metric.compute().item()