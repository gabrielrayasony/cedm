from omegaconf import DictConfig

#-------------------------------------------------------------------------
# Get the experiment name from the configuration
#-------------------------------------------------------------------------
def get_experiment_name(config: DictConfig) -> str:
    """Get the experiment name from the configuration."""
    name = f"{config.model.name}_{config.dataset.name}"
    if config.train.post_training:
        name += "_post_training"
    if config.precond.name:
        name += f"_precond_{config.precond.name}"
    if config.loss.name:
        name += f"_loss_{config.loss.name}"
    if config.train.entropic_sampler:
        name += f"_entropic_sampler"
    return name
