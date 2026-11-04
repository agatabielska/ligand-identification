"""Tune sparse models on train/val, then refit the selected models on train+val."""

import argparse
import contextlib
import gc
import json
import math
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import MinkowskiEngine as ME
import numpy as np
import optuna
import torch
import wandb
import yaml
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

import logger_wandb as log
import models
from data_new import BlobDataset, collation_fn_sparse


ROOT = Path(__file__).resolve().parents[1]
NETWORK_HPARAMETERS = {
    "planes",
    "layers",
    "num_top_down",
    "conv0_kernel_size",
    "feature_size",
    "block",
    "pooling",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(stream.isatty() for stream in self.streams)

    def fileno(self):
        for stream in self.streams:
            if stream.isatty():
                return stream.fileno()
        return self.streams[0].fileno()


def append_error(error_path: Path, message: str) -> None:
    with open(error_path, "a", buffering=1) as error_file:
        error_file.write(message)
        if not message.endswith("\n"):
            error_file.write("\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def resolve_config_path(path: str, config_path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(config_path).resolve().parent / candidate


def class_mapping(dataset_path: Path) -> Dict[str, int]:
    names = set()
    for split in ("train", "val", "test", "holdout"):
        split_path = dataset_path / split
        if split_path.is_dir():
            names.update(d.name for d in split_path.iterdir() if d.is_dir())
    if not names:
        raise FileNotFoundError(f"No class directories found below {dataset_path}")
    return {name: index for index, name in enumerate(sorted(names))}


def load_split(dataset_path: Path, split: str, labels: Mapping[str, int]):
    path = dataset_path / split
    if not path.is_dir():
        raise FileNotFoundError(f"Missing split '{split}' in {dataset_path}")
    return BlobDataset(str(path), label_to_idx=dict(labels))


def loader(dataset, config: Mapping[str, object], shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    batch_size = int(config["batch_size"])
    drop_last = shuffle
    if drop_last and len(dataset) < batch_size:
        raise ValueError(
            f"Training split has {len(dataset)} samples, but batch_size={batch_size}; "
            "reduce the batch-size search range."
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collation_fn_sparse,
        num_workers=int(config["num_workers"]),
        shuffle=shuffle,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def evaluate(model, data_loader, criterion, device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    loss_total = 0.0
    count = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for coords, features, labels in data_loader:
            batch = ME.SparseTensor(features, coords, device=device)
            target = labels.long().to(device)
            logits = model(batch)
            loss = criterion(logits, target)
            loss_total += loss.item() * target.numel()
            count += target.numel()
            predictions.append(logits.cpu())
            targets.append(target.cpu())
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    predictions_tensor = torch.cat(predictions)
    targets_tensor = torch.cat(targets)
    accuracy = (predictions_tensor.argmax(dim=1) == targets_tensor).float().mean().item()
    return loss_total / count, accuracy, predictions_tensor, targets_tensor


def train(
    train_data,
    validation_data: Optional[object],
    config: Mapping[str, object],
    epochs: int,
    seed: int,
    device: torch.device,
    run,
    checkpoint: Path,
    trial=None,
    progress_label: str = "training",
) -> Dict[str, float]:
    train_loader = loader(train_data, config, shuffle=True, seed=seed)
    validation_loader = (
        loader(validation_data, config, shuffle=False, seed=seed)
        if validation_data is not None
        else None
    )
    model = models.create(
        str(config["model"]),
        overrides={
            key: config[key] for key in NETWORK_HPARAMETERS if key in config
        },
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    log.log_config(
        run=run,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataset=train_data,
    )
    best_loss = math.inf
    best_accuracy = 0.0
    stale_epochs = 0

    epoch_progress = tqdm(
        range(epochs),
        desc=progress_label,
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        train_target_count = 0
        train_predictions = []
        train_targets = []
        batch_progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )
        for batch_index, (coords, features, labels) in batch_progress:
            batch = ME.SparseTensor(features, coords, device=device)
            target = labels.long().to(device)
            logits = model(batch)
            batch_loss = criterion(logits, target)
            loss = batch_loss / int(config["accum_iter"])
            loss.backward()
            train_loss_total += batch_loss.item() * target.numel()
            train_target_count += target.numel()
            train_predictions.append(logits.detach().cpu())
            train_targets.append(target.detach().cpu())
            if (batch_index + 1) % int(config["accum_iter"]) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                run.log(
                    {
                        "train/batch_loss": loss.item() * int(config["accum_iter"]),
                        "train/epoch": epoch,
                        "train/batch": batch_index,
                    }
                )
        if len(train_loader) % int(config["accum_iter"]):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        train_predictions = torch.cat(train_predictions)
        train_targets = torch.cat(train_targets)
        train_loss = train_loss_total / max(train_target_count, 1)
        log.log_epoch(
            run,
            preds=train_predictions,
            target=train_targets,
            epoch=epoch,
            prefix="train",
        )
        run.log(
            {
                "train/loss": train_loss,
                "train/epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        metrics = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        if validation_loader is not None:
            validation_loss, validation_accuracy, predictions, targets = evaluate(
                model, validation_loader, criterion, device
            )
            metrics.update({"validation/loss": validation_loss})
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_accuracy = validation_accuracy
                stale_epochs = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale_epochs += 1
                if stale_epochs >= int(config["patience"]):
                    break
            log.log_epoch(
                run,
                preds=predictions,
                target=targets,
                epoch=epoch,
                prefix="validation",
            )
            run.log(
                {
                    "validation/loss": validation_loss,
                    "validation/epoch": epoch,
                }
            )
            if trial is not None:
                trial.report(validation_accuracy, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            epoch_progress.set_postfix(
                validation_loss=f"{validation_loss:.4f}",
                validation_accuracy=f"{validation_accuracy:.4f}",
            )
        else:
            torch.save(model.state_dict(), checkpoint)
        run.log(metrics)
    return {"best_validation_loss": best_loss, "best_validation_accuracy": best_accuracy}


def suggest_parameters(
    trial: optuna.Trial,
    search: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    result = {}
    for name, spec in search.items():
        if spec["type"] == "choice":
            values = list(spec["values"])
            if values and isinstance(values[0], list):
                choice = trial.suggest_int(f"{name}_index", 0, len(values) - 1)
                result[name] = values[choice]
            else:
                result[name] = trial.suggest_categorical(name, values)
        elif spec["type"] == "log_uniform":
            result[name] = trial.suggest_float(
                name,
                float(spec["min"]),
                float(spec["max"]),
                log=True,
            )
        else:
            raise ValueError(f"Unsupported search type for {name}: {spec['type']}")
    return result


def dataset_config(base: Mapping[str, object], params: Mapping[str, object]) -> Dict[str, object]:
    config = dict(base)
    config.update(params)
    config["model"] = "MinkLoc3D"
    return config


def log_checkpoint_artifact(run, checkpoint: Path, dataset_name: str, stage: str) -> None:
    artifact = wandb.Artifact(
        name=f"{stage}-{dataset_name}-{run.id}",
        type="model",
        metadata={"dataset": dataset_name, "stage": stage},
    )
    artifact.add_file(str(checkpoint))
    run.log_artifact(artifact)


def run_tuning(
    dataset,
    base_config,
    output_dir: Path,
    project: str,
    entity: str,
    error_path: Path,
):
    dataset_path = resolve_path(dataset["path"])
    labels = class_mapping(dataset_path)
    train_data = load_split(dataset_path, "train", labels)
    validation_data = load_split(dataset_path, "val", labels)
    storage = f"sqlite:///{(output_dir / 'optuna.db').resolve()}"
    study = optuna.create_study(
        study_name=f"hpo-{dataset['name']}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(base_config["seed"])),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )

    def objective(trial):
        params = suggest_parameters(trial, base_config["search"])
        run = None
        trial_config = dataset_config(
            {**base_config, "num_workers": base_config["num_workers"]}, params
        )
        checkpoint = output_dir / dataset["name"] / f"trial-{trial.number:03d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        try:
            run = wandb.init(
                project=project,
                entity=entity or None,
                name=f"hpo-{dataset['name']}-trial-{trial.number:03d}",
                group=f"hpo-{dataset['name']}",
                job_type="hyperparameter-search",
                tags=["hpo", dataset["name"]],
                config={"dataset": dataset["name"], "split": "train-val", **params},
                settings=wandb.Settings(init_timeout=180),
                reinit=True,
            )
            metrics = train(
                train_data,
                validation_data,
                trial_config,
                int(base_config["trial_epochs"]),
                int(base_config["seed"]) + trial.number,
                torch.device(
                    "cuda"
                    if torch.cuda.is_available() and base_config["device"] != "cpu"
                    else "cpu"
                ),
                run,
                checkpoint,
                trial=trial,
                progress_label=f"trial {trial.number} training",
            )
            log_checkpoint_artifact(run, checkpoint, dataset["name"], "hpo")
            run.summary.update(metrics)
            trial.set_user_attr("best_validation_loss", metrics["best_validation_loss"])
            return metrics["best_validation_accuracy"]
        except optuna.TrialPruned:
            if run is not None:
                run.summary["pruned"] = True
            raise
        except Exception:
            append_error(
                error_path,
                "\n".join(
                    [
                        f"\n[{datetime.now().isoformat()}] Trial failure",
                        f"dataset={dataset['name']}",
                        f"trial={trial.number}",
                        traceback.format_exc(),
                        "-" * 80,
                    ]
                ),
            )
            raise
        finally:
            if run is not None:
                run.finish()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finished_trials = sum(
        trial.state in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        }
        for trial in study.trials
    )
    remaining_trials = max(0, int(base_config["trials"]) - finished_trials)
    trial_progress = tqdm(
        total=int(base_config["trials"]),
        initial=finished_trials,
        desc=f"{dataset['name']} trials",
        unit="trial",
    )
    if remaining_trials:
        def update_trial_progress(study, trial):
            trial_progress.update(1)
            trial_progress.set_postfix(
                complete=sum(
                    item.state == optuna.trial.TrialState.COMPLETE
                    for item in study.trials
                ),
                pruned=sum(
                    item.state == optuna.trial.TrialState.PRUNED
                    for item in study.trials
                ),
                failed=sum(
                    item.state == optuna.trial.TrialState.FAIL
                    for item in study.trials
                ),
            )

        study.optimize(
            objective,
            n_trials=remaining_trials,
            catch=(Exception,),
            callbacks=[update_trial_progress],
        )
    trial_progress.close()
    if not any(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    ):
        raise RuntimeError(
            f"No completed Optuna trials for {dataset['name']}. "
            f"Inspect logs/hpo/optuna.db and rerun after fixing the recorded failure."
        )
    best_trial = study.best_trial
    best_params = suggest_parameters_from_trial(best_trial, base_config["search"])
    return {
        **best_params,
        "best_validation_loss": best_trial.user_attrs.get(
            "best_validation_loss", float("nan")
        ),
        "best_validation_accuracy": best_trial.value,
        "trial": best_trial.number,
        "study_name": study.study_name,
        "completed_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        ),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in study.trials
        ),
        "failed_trials": sum(
            trial.state == optuna.trial.TrialState.FAIL
            for trial in study.trials
        ),
    }


def write_json_atomically(path: Path, value: Mapping[str, object]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary_path, "w") as results_file:
        json.dump(value, results_file, indent=2)
        results_file.flush()
    temporary_path.replace(path)


def suggest_parameters_from_trial(
    trial: optuna.trial.FrozenTrial,
    search: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    params = {}
    for name, spec in search.items():
        if spec["type"] == "choice":
            values = list(spec["values"])
            if values and isinstance(values[0], list):
                params[name] = values[trial.params[f"{name}_index"]]
            else:
                params[name] = trial.params[name]
        else:
            params[name] = trial.params[name]
    return params


def run_final(dataset, base_config, best, output_dir: Path, project: str, entity: str):
    dataset_path = resolve_path(dataset["path"])
    labels = class_mapping(dataset_path)
    train_data = ConcatDataset(
        [load_split(dataset_path, "train", labels), load_split(dataset_path, "val", labels)]
    )
    test_split = "test" if (dataset_path / "test").is_dir() else "holdout"
    test_data = load_split(dataset_path, test_split, labels)
    config = dataset_config(base_config, best)
    run = wandb.init(
        project=project,
        entity=entity or None,
        name=f"final-{dataset['name']}",
        group=f"final-{dataset['name']}",
        job_type="final-training",
        tags=["final", dataset["name"]],
        config={"dataset": dataset["name"], "split": f"train+val-{test_split}", **best},
        settings=wandb.Settings(init_timeout=180),
        reinit=True,
    )
    checkpoint = output_dir / dataset["name"] / "final.pt"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and base_config["device"] != "cpu" else "cpu"
    )
    train(
        train_data,
        None,
        config,
        int(base_config["final_epochs"]),
        int(base_config["seed"]),
        device,
        run,
        checkpoint,
        progress_label=f"{dataset['name']} final training",
    )
    log_checkpoint_artifact(run, checkpoint, dataset["name"], "final")
    model = models.create(
        str(config["model"]),
        overrides={
            key: config[key] for key in NETWORK_HPARAMETERS if key in config
        },
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    test_loss, test_accuracy, predictions, targets = evaluate(
        model, loader(test_data, config, shuffle=False, seed=int(base_config["seed"])), criterion, device
    )
    run.log({"test/loss": test_loss})
    log.log_epoch(
        run,
        preds=predictions,
        target=targets,
        epoch=-1,
        prefix="test",
    )
    run.summary.update({"test_loss": test_loss, "test_accuracy": test_accuracy})
    run.finish()
    return {"test_loss": test_loss, "test_accuracy": test_accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "cfg" / "tune_sparse.yaml"))
    parser.add_argument("--stage", choices=("tune", "final", "all"), default="all")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    output_dir = resolve_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tuning.log"
    error_path = output_dir / "errors.log"
    with open(log_path, "a", buffering=1) as log_file:
        tee = Tee(sys.__stdout__, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"\n[{datetime.now().isoformat()}] Starting stage: {args.stage}")
            config["datasets"] = [
                {
                    **dataset,
                    "path": str(resolve_config_path(dataset["path"], args.config)),
                }
                for dataset in config["datasets"]
            ]
            seed_everything(int(config["seed"]))
            project = str(config["project"])
            entity = str(config.get("entity", ""))
            results_path = output_dir / "best_hyperparameters.json"
            results = {}

            if args.stage in ("tune", "all"):
                dataset_progress = tqdm(
                    config["datasets"],
                    desc="tuning datasets",
                    unit="dataset",
                )
                for dataset in dataset_progress:
                    dataset_progress.set_postfix(dataset=dataset["name"])
                    results[dataset["name"]] = run_tuning(
                        dataset, config, output_dir, project, entity, error_path
                    )
                    write_json_atomically(results_path, results)
                dataset_progress.close()
            if args.stage == "final":
                with open(results_path) as results_file:
                    results = json.load(results_file)
            if args.stage in ("final", "all"):
                for dataset in tqdm(config["datasets"], desc="final datasets", unit="dataset"):
                    final_metrics = run_final(
                        dataset, config, results[dataset["name"]], output_dir, project, entity
                    )
                    results[dataset["name"]]["final"] = final_metrics
                write_json_atomically(results_path, results)
            print(f"[{datetime.now().isoformat()}] Finished stage: {args.stage}")


if __name__ == "__main__":
    main()
