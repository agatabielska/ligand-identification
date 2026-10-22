"""Train the historical protocol with 12 scalars from complete raw NPZ blobs."""
import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.experiments.spectral_experiments import run_default_from_config
from src.scripts.train import check_config, validate_data_dir
from src.utils.set_seed import set_seed


@hydra.main(config_path="../../configs", config_name="config_raw_npz_scalars", version_base=None)
def main(cfg: DictConfig):
    check_config(cfg)
    validate_data_dir(Path(cfg.paths.data_dir))
    set_seed(cfg.random_seed)
    run_default_from_config(cfg, globals_mode="raw_npz")


if __name__ == "__main__":
    main()
