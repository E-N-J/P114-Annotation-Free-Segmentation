def get_model(model_name, **kwargs):
    if model_name == 'RDA':
        from .rda import RobustDeepAutoencoder
        return RobustDeepAutoencoder(**kwargs)
    elif model_name == 'RPCA':
        from .rpca import RobustPCA
        return RobustPCA(**kwargs)
    elif model_name == 'ceVAE':
        from .cevae import ContextEncodingVAE
        return ContextEncodingVAE(**kwargs)
    else:
        raise ValueError(f"Model '{model_name}' not recognised.")
