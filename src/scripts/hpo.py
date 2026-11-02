"""Optuna hyperparameter search for the Lightning/Hydra spectral model.

Stages
------
tune  : run Optuna trials (train on train split, objective measured on val split).
final : retrain on train+val with the best params for a fixed number of epochs (the best
        epoch found while tuning), then evaluate once on the test split.
all   : tune, then final.

Usage (from repo root, same as normal training; trailing args are Hydra overrides):

    uv run -m src.scripts.hpo --n-trials 40 --stage all --search configs/hpo/spectral.yaml \
        paths.data_dir=data/unthresholded_transform_probabilistic model=spectral

The search space (which parameters, which intervals) lives in the --search YAML file.

Parallel workers: start the same command in several processes (one per GPU, e.g. with
CUDA_VISIBLE_DEVICES) -- they share the SQLite study. For heavy parallelism use a
Postgres/MySQL URL via --storage.
"""

import argparse
import gc
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import pytorch_lightning as pl
import torch
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

import src.scripts.train as train_module
from src.pipeline.data_loader import validate_data_dir
from src.scripts.train import TRAINING_BUILDERS
from src.utils.set_seed import set_seed

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

Search = dict[str, dict[str, Any]]
TrialState = optuna.trial.TrialState


# --------------------------------------------------------------------------- #
# Data cache
# --------------------------------------------------------------------------- #
def _install_data_cache() -> None:
    """Load the data once per process instead of once per trial.

    train.py's builders look up `build_data` as a module global, so wrapping it here makes
    every trial (and the final stage) reuse the same loaders. The cache key covers
    everything that can change the data, so searching over data.* still works.
    """
    original = train_module.build_data
    cache: dict[tuple, Any] = {}

    def cached(cfg, kind):
        key = (
            kind,
            str(cfg.paths.data_dir),
            OmegaConf.to_yaml(cfg.data),
            str(cfg.train.get("normalize_data")),
        )
        if key not in cache:
            cache[key] = original(cfg, kind)
        return cache[key]

    train_module.build_data = cached


_install_data_cache()


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def make_cfg(base_overrides: list[str], extra: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=[*base_overrides, *extra])


def _fmt(value: Any) -> str:
    """Format a Python value as a Hydra override value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt(v) for v in value) + "]"
    return str(value)


def params_to_overrides(params: dict[str, Any], search: Search | None = None) -> list[str]:
    """{config path: value} -> Hydra override strings (plus any `also` targets)."""
    search = search or {}
    overrides = []
    for path, value in params.items():
        targets = [path, *search.get(path, {}).get("also", [])]
        overrides += [f"{target}={_fmt(value)}" for target in targets]
    return overrides


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def load_search(path: str) -> Search:
    """Read and validate the search-space YAML, dropping disabled parameters."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    search: Search = {}
    for name, spec in raw.items():
        if spec.get("enabled", True) is False:
            continue
        kind = spec.get("type")
        if kind in ("float", "int"):
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"{name}: '{kind}' needs low and high")
        elif kind == "categorical":
            if not spec.get("values"):
                raise ValueError(f"{name}: categorical needs a non-empty 'values' list")
        else:
            raise ValueError(f"{name}: unknown type {kind!r} (float | int | categorical)")
        search[name] = spec
    if not search:
        raise ValueError(f"No enabled parameters in {path}")
    return search


def check_search_against_config(search: Search, cfg: DictConfig, search_path: str) -> None:
    """Make sure every searched path exists in the config; prune missing `also` targets."""
    missing = object()

    def exists(path: str) -> bool:
        return OmegaConf.select(cfg, path, default=missing) is not missing

    for name, spec in search.items():
        if not exists(name):
            raise KeyError(f"Search parameter '{name}' does not exist in the Hydra config")
        # `also` targets are optional extras: skip the ones this config doesn't have
        present = []
        for path in spec.get("also", []):
            if exists(path):
                present.append(path)
            else:
                print(f"Note: '{path}' (also for '{name}') not in config, skipping it")
        spec["also"] = present
    print(f"Tuning {len(search)} parameters from {search_path}: {', '.join(search)}")


def _is_complex(spec: dict[str, Any]) -> bool:
    """True if a categorical has list/dict values (Optuna needs those stored as JSON)."""
    return any(isinstance(v, (list, dict)) for v in spec["values"])


def decode_params(search: Search, raw: dict[str, Any]) -> dict[str, Any]:
    """Optuna params (lists stored as JSON strings) -> real values keyed by config path."""
    out = {}
    for name, spec in search.items():
        if name not in raw:
            continue
        value = raw[name]
        if spec["type"] == "categorical" and _is_complex(spec):
            value = json.loads(value)
        out[name] = value
    return out


def suggest_parameters(trial: optuna.Trial, search: Search) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for name, spec in search.items():
        kind = spec["type"]

        if kind == "categorical":
            values = spec["values"]
            if _is_complex(spec):
                values = [json.dumps(v) for v in values]
            raw[name] = trial.suggest_categorical(name, values)
            continue

        low, high = spec["low"], spec["high"]
        cap_by = spec.get("cap_by")
        if cap_by in raw:  # the capping parameter was already sampled this trial
            cap = decode_params(search, raw)[cap_by]
            high = max(low, min(high, cap))

        log = bool(spec.get("log", False))
        step = spec.get("step")
        if kind == "int":
            raw[name] = trial.suggest_int(name, int(low), int(high), step=int(step or 1), log=log)
        else:
            raw[name] = trial.suggest_float(
                name, float(low), float(high),
                step=None if step is None else float(step),
                log=log,
            )
    return decode_params(search, raw)


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _mode(direction: str) -> str:
    return "max" if direction == "maximize" else "min"


def _log_error(path: Path, study: str, trial: int, text: str) -> None:
    with open(path, "a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] study={study} trial={trial}\n{text}\n{'-' * 80}\n")


def _finish_wandb(args) -> None:
    if not args.no_wandb and wandb.run is not None:
        wandb.finish()


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Pruning callback (no optuna-integration dependency)
# --------------------------------------------------------------------------- #
class OptunaPruning(pl.Callback):
    """Reports the monitored metric to Optuna each epoch and tracks the best value/epoch."""

    def __init__(self, trial: optuna.Trial, monitor: str, direction: str):
        self.trial = trial
        self.monitor = monitor
        self.maximize = direction == "maximize"
        self.best = -math.inf if self.maximize else math.inf
        self.best_epoch = 0

    def _improved(self, value: float) -> bool:
        return value > self.best if self.maximize else value < self.best

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        value = float(value)
        if not math.isfinite(value):
            raise optuna.TrialPruned(f"Non-finite {self.monitor}")

        if self._improved(value):
            self.best, self.best_epoch = value, trainer.current_epoch
        self.trial.report(value, trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# --------------------------------------------------------------------------- #
# Stage 1: tune
# --------------------------------------------------------------------------- #
def make_objective(args, search: Search, base_overrides: list[str], error_path: Path):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(trial, search)
        cfg = make_cfg(
            base_overrides,
            [
                *params_to_overrides(params, search),
                f"train.epochs={args.trial_epochs}",
                f"random_seed={args.seed + trial.number}",
            ],
        )
        set_seed(cfg.random_seed)

        pruning = OptunaPruning(trial, args.metric, args.direction)
        try:
            model, (train_loader, val_loader, _) = TRAINING_BUILDERS[cfg.model.type](cfg)

            logger = False
            if not args.no_wandb:
                logger = WandbLogger(
                    project=args.project,
                    name=f"{args.study_name}-trial-{trial.number:03d}",
                    group=args.study_name,
                    job_type="hpo",
                    tags=["hpo"],
                    config={**params, "trial": trial.number},
                )

            trainer = pl.Trainer(
                max_epochs=cfg.train.epochs,
                callbacks=[
                    EarlyStopping(
                        monitor=args.metric,
                        mode=_mode(args.direction),
                        patience=cfg.train.early_stopping.patience,
                    ),
                    pruning,
                ],
                accelerator="auto",
                devices=cfg.machine.devices,
                precision=cfg.train.get("precision", "32-true"),
                log_every_n_steps=10,
                logger=logger,
                gradient_clip_val=cfg.train.gradient_clip_val,
                enable_checkpointing=False,  # final stage retrains; no need to store trials
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(model, train_loader, val_loader)

            if not math.isfinite(pruning.best):
                raise optuna.TrialPruned("Metric never logged / not finite")
            trial.set_user_attr("best_epoch", pruning.best_epoch)
            return pruning.best

        except optuna.TrialPruned:
            raise
        except torch.cuda.OutOfMemoryError:
            # config too large for the GPU -> treat as a bad trial, not a crash
            _log_error(error_path, args.study_name, trial.number, "CUDA OOM\n" + traceback.format_exc())
            raise optuna.TrialPruned("CUDA OOM")
        except Exception:
            _log_error(error_path, args.study_name, trial.number, traceback.format_exc())
            raise
        finally:
            _finish_wandb(args)
            _free_memory()

    return objective


def _create_study(args, output_dir: Path) -> optuna.Study:
    return optuna.create_study(
        study_name=args.study_name,
        storage=args.storage or f"sqlite:///{(output_dir / 'optuna.db').resolve()}",
        load_if_exists=True,
        direction=args.direction,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
    )


def _count(study: optuna.Study, *states: TrialState) -> int:
    return sum(t.state in states for t in study.trials)


def run_tune(args, search: Search, base_overrides: list[str], output_dir: Path) -> dict[str, Any]:
    study = _create_study(args, output_dir)

    finished = _count(study, TrialState.COMPLETE, TrialState.PRUNED)
    remaining = max(0, args.n_trials - finished)
    print(f"Study '{study.study_name}': {finished} finished, running {remaining} more")

    if remaining:
        study.optimize(
            make_objective(args, search, base_overrides, output_dir / "errors.log"),
            n_trials=remaining,
            catch=(Exception,),
        )

    n_complete = _count(study, TrialState.COMPLETE)
    if not n_complete:
        raise RuntimeError(f"No completed trials. See {output_dir / 'errors.log'}")

    best = study.best_trial
    result = {
        "study_name": study.study_name,
        "metric": args.metric,
        "best_value": best.value,
        "best_trial": best.number,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "params": decode_params(search, best.params),
        "n_complete": n_complete,
        "n_pruned": _count(study, TrialState.PRUNED),
        "n_failed": _count(study, TrialState.FAIL),
    }

    # write atomically so parallel readers never see a half-written file
    tmp = output_dir / "best_hyperparameters.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(output_dir / "best_hyperparameters.json")
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- #
# Stage 2: final retrain on train+val
# --------------------------------------------------------------------------- #
class MergedTensorLoader:
    """Minimal GPULoader-style loader over concatenated (x, y, g) tensors.

    Yields shuffled (x, y, g) batches each epoch, drops the last incomplete batch
    (BatchNorm dislikes tiny batches) and exposes .x/.y/.g like the loaders it replaces.
    """

    def __init__(self, x, y, g, batch_size: int):
        self.x, self.y, self.g = x, y, g
        self.batch_size = batch_size

    def __len__(self):
        return len(self.y) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.y), device=self.y.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.x[idx], self.y[idx], self.g[idx]


_TENSOR_ATTRS = {
    "x": ("x", "X", "data", "points", "coords"),
    "y": ("y", "labels"),
    "g": ("g", "globals"),
}


def _loader_tensors(loader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull (x, y, g) out of a tensor-holding loader (e.g. GPULoader)."""
    found = {}
    for key, candidates in _TENSOR_ATTRS.items():
        attr = next((c for c in candidates if torch.is_tensor(getattr(loader, c, None))), None)
        if attr is None:
            raise AttributeError(
                f"{type(loader).__name__} has no tensor for '{key}' "
                f"(tried {candidates}). Its attributes: {sorted(vars(loader))}."
            )
        found[key] = getattr(loader, attr)
    return found["x"], found["y"], found["g"]


def _merge_torch_loaders(train_loader: DataLoader, val_loader: DataLoader) -> DataLoader:
    return DataLoader(
        ConcatDataset([train_loader.dataset, val_loader.dataset]),
        batch_size=train_loader.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
        persistent_workers=train_loader.num_workers > 0 and train_loader.persistent_workers,
    )


def _merge_tensor_loaders(train_loader, val_loader) -> MergedTensorLoader:
    xt, yt, gt = _loader_tensors(train_loader)
    xv, yv, gv = _loader_tensors(val_loader)
    device = xt.device

    x = torch.cat([xt, xv.to(device)])
    y = torch.cat([yt.view(-1), yv.view(-1).to(device)])
    g = torch.cat([gt, gv.to(device)])

    batch_size = getattr(train_loader, "batch_size", None) or len(next(iter(train_loader))[1])
    print(f"Merged train+val: {len(y)} samples, batch_size={batch_size}, device={device}")
    return MergedTensorLoader(x, y, g, int(batch_size))


def merge_loaders(train_loader, val_loader):
    """Merge train and val into one training loader."""
    if isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader):
        return _merge_torch_loaders(train_loader, val_loader)
    return _merge_tensor_loaders(train_loader, val_loader)


def refit_spectral_statistics(model, loader) -> None:
    """Recompute class counts / global-feature stats on the merged train+val data.

    build_spectral_training computes them on train only. Batches are (x, y, global_values).
    """
    if hasattr(loader, "y") and hasattr(loader, "g"):
        y, g = loader.y.view(-1), loader.g
    else:
        batches = list(loader)
        y = torch.cat([b[1].view(-1) for b in batches])
        g = torch.cat([b[2] for b in batches])

    model.set_class_counts(torch.bincount(y, minlength=model.num_classes))
    model.global_mean.copy_(g.mean(0))
    model.global_std.copy_(g.std(0).clamp_min(1e-6))


def _final_epochs(args, best: dict[str, Any]) -> int:
    """Explicit --final-epochs, else the best tuning epoch (0-indexed) + 1, else trial_epochs."""
    if args.final_epochs:
        return args.final_epochs
    best_epoch = best.get("best_epoch")
    return args.trial_epochs if best_epoch is None else best_epoch + 1


def run_final(args, search: Search, base_overrides: list[str], output_dir: Path) -> None:
    best = json.loads((output_dir / "best_hyperparameters.json").read_text())
    epochs = _final_epochs(args, best)

    cfg = make_cfg(
        base_overrides,
        [
            *params_to_overrides(best["params"], search),
            f"train.epochs={epochs}",
            f"random_seed={args.seed}",
        ],
    )
    print(f"Final config ({epochs} epochs, train+val):\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.random_seed)
    validate_data_dir(Path(cfg.paths.data_dir))

    model, (train_loader, val_loader, test_loader) = TRAINING_BUILDERS[cfg.model.type](cfg)
    merged_loader = merge_loaders(train_loader, val_loader)
    if cfg.model.type == "spectral":
        refit_spectral_statistics(model, merged_loader)

    run_name = f"final-{args.study_name}"
    checkpoint_dir = Path(cfg.paths.model_checkpoint) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = False
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.project,
            name=run_name,
            group=args.study_name,
            job_type="final",
            tags=["final"],
            config={**OmegaConf.to_container(cfg, resolve=True), "final_epochs": epochs},
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            # no validation split -> no monitor; keeps the last epoch
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="final-{epoch:02d}",
                save_top_k=1,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        accelerator="auto",
        devices=cfg.machine.devices,
        precision=cfg.train.get("precision", "32-true"),
        log_every_n_steps=10,
        logger=logger,
        gradient_clip_val=cfg.train.gradient_clip_val,
    )
    trainer.fit(model, merged_loader)  # no val loader: no early stopping / selection
    trainer.test(model, dataloaders=test_loader)  # last-epoch weights, test touched once
    _finish_wandb(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search", default=str(CONFIG_DIR / "hpo" / "spectral.yaml"),
                   help="YAML file defining the parameters and intervals to tune")
    p.add_argument("--stage", choices=("tune", "final", "all"), default="all")
    p.add_argument("--n-trials", type=int, default=100, help="Total trials in the study")
    p.add_argument("--trial-epochs", type=int, default=25)
    p.add_argument("--final-epochs", type=int, default=None)
    p.add_argument("--metric", default="val_acc", help="Logged metric to optimise")
    p.add_argument("--direction", choices=("minimize", "maximize"), default="maximize")
    p.add_argument("--study-name", default="hpo-spectral")
    p.add_argument("--storage", default=None, help="Optuna storage URL (default: sqlite in output dir)")
    p.add_argument("--output-dir", default="logs/hpo")
    p.add_argument("--project", default="ligand-identification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. paths.data_dir=...")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate the base config, data dir and search space once, up front.
    base_cfg = make_cfg(args.overrides, [])
    validate_data_dir(Path(base_cfg.paths.data_dir))
    search = load_search(args.search)
    check_search_against_config(search, base_cfg, args.search)

    if args.stage in ("tune", "all"):
        run_tune(args, search, args.overrides, output_dir)
    if args.stage in ("final", "all"):
        run_final(args, search, args.overrides, output_dir)


if __name__ == "__main__":
    main()