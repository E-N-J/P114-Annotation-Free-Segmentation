import os
import torch
import sys

def save_rpca_results(x, L, S, results_root="./results"):
    os.makedirs(results_root, exist_ok=True)
    save_path = os.path.join(results_root, "rpca_results.pt")
    torch.save({'L': L.cpu(), 'S': S.cpu(), 'X': x.cpu()}, save_path)
    print(f"RPCA results saved to {save_path}.")

def get_environment():
    """
    Returns a string indicating the environment: 'colab', 'notebook', or 'script'.
    """
    if 'google.colab' in sys.modules:
        return 'colab'
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'notebook'
        if 'terminal' in ipy_str:
            return 'script'
    except NameError:
        # get_ipython() is not defined, so we are in standard Python
        return 'script'
        
    return 'script'
