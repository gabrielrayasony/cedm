from typing import List
from omegaconf import DictConfig
import lightning as L
from pathlib import Path
from callbacks.entropy_callback import EntropyCallback
from callbacks.sampling_callback import SamplingCallback


def get_callbacks(cfg: DictConfig, results_dir: Path) -> List[L.Callback]:
    """Get list of callbacks for training.
    
    Args:
        cfg: Hydra configuration object
        results_dir: Directory to save results
        
    Returns:
        List of Lightning callbacks
    """
    callbacks = []

    # Model checkpointing
    callbacks.append(
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=results_dir / "checkpoints",
            monitor=cfg.logging.monitor,
            mode=cfg.logging.mode,
            save_top_k=cfg.logging.save_top_k
        )
    )

    # Sampling callback
    callbacks.append(
        SamplingCallback(
            sampling_config=cfg.sampling,
            viz_config=cfg.viz,
            save_dir=results_dir / "samples",
            sampling_interval=cfg.train.sampling_interval
        )
    )

    if cfg.train.use_entropic_sampler:
        # Entropy callback
        callbacks.append(
            EntropyCallback(
                interval=cfg.train.entropy_interval,
                num_timesteps=cfg.train.entropy_num_timesteps,
                save_dir=results_dir / "entropy",
                switch_to_entropic=cfg.train.use_entropic_sampler,
                switch_epoch=cfg.train.entropic_warmup_epochs,
                entropic_config={
                    "sigma_min": cfg.precond.sigma_min,
                    "sigma_max": cfg.precond.sigma_max,
                    "num_grid_points": cfg.train.entropic_num_grid_points
                }
            )
        )
    
    return callbacks