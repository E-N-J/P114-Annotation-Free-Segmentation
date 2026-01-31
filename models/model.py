from .rda import RobustDeepAutoencoder
from .rpca import RobustPCA

def get_model(model_name, **kwargs):
    if model_name == 'RDA':
        return RobustDeepAutoencoder(**kwargs)
    elif model_name == 'RPCA':
        return RobustPCA(**kwargs)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")
