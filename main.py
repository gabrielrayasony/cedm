import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from pathlib import Path
from typing import List

from utils.model_utils import get_model
from models.lightning.edm_lightning import EDMLightning
from utils.data_utils import get_datamodule, rescaling_inv
import logging
from utils.plots import plot_data
from utils.callback_utils import get_callbacks

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
    model, _ = get_model(cfg)

    # Get callbacks
    callbacks = get_callbacks(cfg, results_dir)

    # Set up loss function based on model type
    logger.info(f"Setting up {cfg.model.name} loss function with {cfg.loss.name.upper()} loss class...")
    
    # Get Training Data
    data_module = get_datamodule(dataset_name=cfg.dataset.name, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    data_module.prepare_data()
    data_module.setup()

    # Plot Training Data
    x, _ = next(iter(data_module.train_dataloader()))
    plot_data(x, name="training_data", figsize=(3, 3), workdir=results_dir)
    del x

    # Set up Lightning module
    logger.info("Setting up Lightning module...")
    lightning_model = EDMLightning(
        model=model,
        config=cfg
    )

    
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