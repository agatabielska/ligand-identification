import os
from datetime import datetime
from typing import Union, List, OrderedDict

import wandb
import torch
import torchmetrics
import yaml

PROJECT = "ligand-identification"
ENTITY = "adam-mazur"


def get_run(tags: Union[List[str], None] = None):
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        mode="online",
        tags=tags if tags is not None else []
    )
    return run


def fetch_state_dict(
        model_name: str,
        run_id: str,
        epoch: int,
) -> OrderedDict[str, torch.Tensor]:
    artifact_name = f"{ENTITY}/{PROJECT}/{model_name}-run{run_id}-epoch{epoch}:latest"
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()
    state_dict = torch.load(os.path.join(artifact_dir, "model.pt"))
    return state_dict


def compute_metrics(preds: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all metrics and return as a flat dict with 'eval/' prefix."""
    target_idx  = target.long()
    num_classes = preds.shape[1]

    top5_acc  = torchmetrics.functional.accuracy(preds, target_idx, top_k=5,  num_classes=num_classes) if num_classes > 5  else 1.0
    top10_acc = torchmetrics.functional.accuracy(preds, target_idx, top_k=10, num_classes=num_classes) if num_classes > 10 else 1.0
    top20_acc = torchmetrics.functional.accuracy(preds, target_idx, top_k=20, num_classes=num_classes) if num_classes > 20 else 1.0

    def to_float(x):
        return x.item() if isinstance(x, torch.Tensor) else float(x)

    return {
        "eval/accuracy":        to_float(torchmetrics.functional.accuracy(preds, target_idx, num_classes=num_classes)),
        "eval/top5_accuracy":   to_float(top5_acc),
        "eval/top10_accuracy":  to_float(top10_acc),
        "eval/top20_accuracy":  to_float(top20_acc),
        "eval/macro_recall":    to_float(torchmetrics.functional.recall(preds, target_idx, average="macro", num_classes=num_classes)),
        "eval/micro_recall":    to_float(torchmetrics.functional.recall(preds, target_idx, average="micro", num_classes=num_classes)),
        "eval/micro_precision": to_float(torchmetrics.functional.precision(preds, target_idx, average="micro", num_classes=num_classes)),
        "eval/micro_f1":        to_float(torchmetrics.functional.f1_score(preds, target_idx, average="micro", num_classes=num_classes)),
    }


def log_epoch_and_model(
        run,
        model: torch.nn.Module,
        epoch: int,
        preds: torch.Tensor,
        target: torch.Tensor,
        model_name: str,
        config_file: str = "../cfg/train.yaml",
) -> None:
    """Save model checkpoint and log all metrics."""
    with open(config_file) as fp:
        config = yaml.safe_load(fp)

    models_path = os.path.join("logs", "models")
    os.makedirs(models_path, exist_ok=True)
    time_str = str(datetime.now()).replace(" ", "-")
    file_path = os.path.join(models_path, f"{time_str}-epoch-{epoch}.pt")
    torch.save(model.state_dict(), file_path)

    artifact = wandb.Artifact(
        name=f"LIGANDS-{config['model']}".upper(),
        type="model",
        metadata={"run_id": run.id, "epoch": epoch}
    )
    artifact.add_file(file_path)
    run.log_artifact(artifact)

    # reuse log_epoch for the actual metric logging
    log_epoch(run, preds, target, epoch, prefix="eval")

    # local log file
    metrics = compute_metrics(preds, target)
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "log.txt")
    if not os.path.isfile(path):
        with open(path, "a") as fp:
            fp.write("time,epoch," + ",".join(k.split("/")[1] for k in metrics.keys()) + "\n")
    time_str = str(datetime.now()).replace(" ", "-")
    with open(path, "a") as fp:
        fp.write(f"{time_str},{epoch}," + ",".join(str(v) for v in metrics.values()) + "\n")


def log_config(
        run,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.nn.Module,
        dataset,
) -> None:
    if hasattr(dataset, 'labels'):
        instances = dataset.labels.shape[0]
        num_labels = dataset.labels.shape[1]
    elif hasattr(dataset, "datasets"):
        instances = sum(len(child) for child in dataset.datasets)
        num_labels = len({
            label
            for child in dataset.datasets
            for _, label in child.samples
        })
    else:
        instances = len(dataset)
        num_labels = len(set(label for _, label in dataset.samples))

    run.config.update({
        "model/class":             type(model).__name__,
        "model/structure":         str(model),
        "dataset/instances":       instances,
        "dataset/labels":          num_labels,
        "criterion/class":         type(criterion).__name__,
        "optimizer/class":         type(optimizer).__name__,
        "optimizer/learning_rate": optimizer.defaults["lr"],
        "optimizer/weight_decay":  optimizer.defaults["weight_decay"],
    })


def log_train_batch(run, loss: float, epoch: int, batch_idx: int) -> None:
    """Log training loss after each optimizer step."""
    run.log({
        "train/loss": loss,
        "train/epoch": epoch,
        "train/batch": batch_idx,
    })

def log_epoch(
        run,
        preds: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        prefix: str = "eval",
) -> None:
    """Log all metrics for a given epoch to wandb."""
    metrics = compute_metrics(preds, target)
    # Optionally remap prefix if different from 'eval'
    if prefix != "eval":
        metrics = {k.replace("eval/", f"{prefix}/", 1): v for k, v in metrics.items()}
    metrics["epoch"] = epoch
    run.log(metrics)