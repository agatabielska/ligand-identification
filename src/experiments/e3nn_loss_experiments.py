"""Train and test the main spectral e3nn model with every supported loss."""

from datetime import datetime, timezone

import hydra
from omegaconf import DictConfig, OmegaConf

from src.experiments.ensemble3models.losses import DEFINED_LOSSES
from src.scripts.train import run_training
from src.utils.set_seed import set_seed


def format_loss_run_name(run_id: str, loss_cfg: dict) -> str:
    """Include the loss name and all explicitly configured parameters."""
    loss_name = str(loss_cfg["name"])
    parameters = [
        f"{key}={value}"
        for key, value in loss_cfg.items()
        if key != "name"
    ]
    return "-".join([run_id, loss_name, *parameters])


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for loss_cfg in DEFINED_LOSSES:
        run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        run_cfg.train.loss = loss_cfg
        run_cfg.wandb.run_name = format_loss_run_name(run_id, loss_cfg)
        run_cfg.wandb.tags = [
            f"loss:{loss_cfg['name']}",
            *[
                f"{key}:{value}"
                for key, value in loss_cfg.items()
                if key != "name"
            ],
        ]
        if loss_cfg["name"] == "focal":
            run_cfg.train.gradient_clip_val = 1.0
        set_seed(int(run_cfg.random_seed))
        print(f"Training and testing loss: {loss_cfg['name']}")
        run_training(run_cfg)


if __name__ == "__main__":
    main()
