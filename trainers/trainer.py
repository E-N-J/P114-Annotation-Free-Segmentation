def get_trainer(trainer_name, model, loader, **kwargs):
    if trainer_name == 'RDA':
        from .rda import RDATrainer
        return RDATrainer(model=model, loader=loader, **kwargs)
    if trainer_name == 'RVAE':
        from .rvae import VAETrainer
        return VAETrainer(model=model, loader=loader, **kwargs)
    else:
        raise ValueError(f"Trainer '{trainer_name}' not recognized.")
