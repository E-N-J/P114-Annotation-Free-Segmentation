import torch
import os
from .utils import save_rpca_results
import torch.nn.functional as F

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
        X_input = X_cache if X_cache is not None else F.interpolate(X_input, size=target_size).cpu()
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

def run_deep_models_inference(X_input, models_dict, target_size=None):
    """Runs inference for deep learning models."""
    print("Running Deep Models Inference...")
    model_results = {}
    device_inputs = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = next(model.buffers()).device
        
            if device not in device_inputs:
                device_inputs[device] = X_input.to(device, non_blocking=True)
            
            X_device = device_inputs[device]
            output = model(X_device)
            if isinstance(output, tuple):
                output = output[0]
            L_device = output
            S_device = X_device - L_device 

            if target_size is not None:
                L_device = F.interpolate(L_device, size=target_size)
                S_device = F.interpolate(S_device, size=target_size)

            model_results[name] = [L_device.cpu(), S_device.cpu()]
            
    print("Inference done.\n")
    return model_results