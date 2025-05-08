import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
import torch
from pathlib import Path

from models.edm import EDM
from utils.model_utils import get_model
from datamodules.toy_datasets import ToyDataModule
from models.lightning.edm_lightning import EDMLightning
from training.losses import get_loss_fn
from callbacks.sampling_callback import SamplingCallback
from utils.data_utils import get_datamodule, rescaling_inv
import logging
from utils.model_utils import get_neural_net
from utils.plots import plot_2d_data, plot_image_grid

# Set up logging
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    """Main training function using Hydra and Lightning.
    
    Args:
        cfg: Hydra configuration object
    """
    # Create experiment directory
    experiment_name = f"{cfg.model.name}_{cfg.dataset.name}"
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed for reproducibility
    L.seed_everything(42)
    
    # Set up model
    logger.info("Setting up model...")
    model = get_model(cfg)

    # Set up loss function based on model type
    logger.info(f"Setting up {cfg.model.name} loss function with {cfg.loss.name.upper()} loss class...")
    loss_fn = get_loss_fn(cfg)
    
    # Get Trainig Noise Scheduler from Loss Function
    # noise_scheduler = loss_fn.noise_scheduler

    # Get Training Data
    data_module = get_datamodule(dataset_name=cfg.dataset.name, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    data_module.prepare_data()
    data_module.setup()
    x, _ = next(iter(data_module.train_dataloader()))
    # Plot Training Data
    if x.ndim == 2:
        plot_2d_data(x.numpy(), name="training_data", figsize=(3, 3), workdir=results_dir)
    else:
        plot_image_grid(rescaling_inv(x[:64].numpy()), name="training_data", figsize=(3, 3), workdir=results_dir)

    # Set up Lightning module
    logger.info("Setting up Lightning module...")
    lightning_model = EDMLightning(
        model=model,
        loss_fn=loss_fn,
        config=cfg
    )
    
    # Set up callbacks
    callbacks = [
        # Model checkpointing
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=results_dir / "checkpoints",
            monitor=cfg.logging.monitor,
            mode=cfg.logging.mode,
            save_top_k=cfg.logging.save_top_k
        ),
        # Sampling callback
        SamplingCallback(
            sampling_config=cfg.sampling,
            viz_config=cfg.viz,
            save_dir=results_dir / "samples",
            sampling_interval=cfg.train.sampling_interval
        )
    ]
    
    # Set up trainer without logger
    logger.info("Setting up trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        callbacks=callbacks,
        enable_progress_bar=True,  # Keep progress bar for development
        logger=False  # Disable logging
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(lightning_model, data_module)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()