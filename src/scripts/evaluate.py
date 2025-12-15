from src.models.clifford.model import CliffordSteerableNetwork
from src.models.e3nn.model import E3NNPointCloudModel
from omegaconf import DictConfig, OmegaConf
from src.utils.set_seed import set_seed
from hydra.utils import to_absolute_path
from pathlib import Path
import pytorch_lightning as pl
from src.scripts.train import (
    get_dataloader,
    get_dataset,
    transform_e3nn,
    transform_clifford,
)
import hydra


def resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_path_cfg = OmegaConf.select(cfg, "paths.checkpoint_path")
    if checkpoint_path_cfg is None:
        raise ValueError(
            "Set paths.checkpoint_path in the config or via CLI to evaluate a model."
        )

    checkpoint_path = Path(checkpoint_path_cfg)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(to_absolute_path(checkpoint_path_cfg))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Please provide a valid path."
        )

    return checkpoint_path


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.random_seed)

    if cfg.model.type == "clifford":
        transform = transform_clifford
    elif cfg.model.type == "e3nn":
        transform = transform_e3nn
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    test_dataset = get_dataset(cfg.paths.test_data, cfg, transform)
    test_dataloader = get_dataloader(test_dataset, cfg)

    checkpoint_path = resolve_checkpoint_path(cfg)

    if cfg.model.type == "clifford":
        model = CliffordSteerableNetwork.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            p=cfg.model.p,
            q=cfg.model.q,
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.train.out_channels,
            n_shells=cfg.model.n_shells,
            kernel_size=cfg.model.kernel_size,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.model.type == "e3nn":
        model = E3NNPointCloudModel.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            num_classes=cfg.train.out_channels,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=cfg.machine.devices,
        log_every_n_steps=10,
    )

    trainer.test(model=model, dataloaders=test_dataloader, verbose=True)


if __name__ == "__main__":
    main()
