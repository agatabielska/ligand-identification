from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.scripts.train import run_training
from src.utils.set_seed import set_seed


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
MODEL_CONFIG = ROOT / "configs" / "model" / "spectral.yaml"
MACHINE_CONFIG = ROOT / "configs" / "machine" / "hpc.yaml"
EXPERIMENT_CONFIG = Path(__file__).with_name("config.yaml")


def build_config(model_override: DictConfig) -> DictConfig:
    """Merge one experiment entry into the default spectral e3nn config."""
    override = OmegaConf.to_container(model_override, resolve=True)
    if not isinstance(override, dict):
        raise ValueError("Each entry in models must be a mapping.")

    name = override.pop("name", None)
    data_path = override.pop("data_path", None)
    num_sample_points = override.pop("num_sample_points", None)
    architecture = override.pop("architecture", "spectral")
    if architecture != "spectral":
        raise ValueError(f"{name or 'Model'} must use architecture=spectral.")

    cfg = OmegaConf.merge(
        OmegaConf.load(DEFAULT_CONFIG),
        {"model": OmegaConf.load(MODEL_CONFIG), "machine": OmegaConf.load(MACHINE_CONFIG)},
    )
    if data_path is not None:
        cfg.paths.data_dir = str((ROOT / data_path).resolve()) if not Path(data_path).is_absolute() else data_path
    if name is not None:
        cfg.wandb.run_name = name

    model_fields = set(OmegaConf.to_container(cfg.model, resolve=True))
    model_values = {key: value for key, value in override.items() if key in model_fields}
    unknown = set(override) - model_fields
    if unknown:
        raise ValueError(f"Unsupported model override(s): {', '.join(sorted(unknown))}")
    tags = [
        f"dataset:{Path(cfg.paths.data_dir).name}",
        f"model:{name or 'unnamed'}",
    ]
    if num_sample_points is not None:
        tags.append(f"points:{num_sample_points}")
    return OmegaConf.merge(cfg, {"model": model_values, "wandb": {"tags": tags}})


@hydra.main(config_path="ensemble3models", config_name="config", version_base=None)
def main(experiment_cfg: DictConfig) -> None:
    run_id = experiment_cfg.get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    print(f"Training ensemble run: {run_id}")
    for model_cfg in experiment_cfg.models:
        cfg = build_config(model_cfg)
        cfg.wandb.project = experiment_cfg.wandb.project
        cfg.wandb.experiment = experiment_cfg.wandb.experiment
        cfg.train.loss = OmegaConf.create(
            OmegaConf.to_container(experiment_cfg.loss, resolve=True)
        )
        cfg.wandb.run_name = f"{run_id}-{cfg.wandb.run_name}"
        cfg.wandb.tags = [f"run:{run_id}", cfg.wandb.experiment, *list(cfg.wandb.tags)]
        set_seed(cfg.random_seed)
        run_training(cfg)


if __name__ == "__main__":
    main()