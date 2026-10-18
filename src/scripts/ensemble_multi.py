"""Ensemble trained runs that may have DIFFERENT architectures (n_mix, n_shells, lmax, ...).

Each model is rebuilt from the hyper-parameters stored inside its own checkpoint, so you only
pass run names. For every run the checkpoint with the lowest val_loss is used.

  uv run -m src.scripts.ensemble_multi paths.data_dir=data/unthresholded_transform_probabilistic \
    model=spectral machine.batch_size=256 machine.num_workers=8 \
    +runs=[spec_v3_reg_h512,spec_v5_M,spec_v5_L]

Runs that cannot be loaded into SpectralModel (e.g. the original v1 model) are skipped.
Tau (post-hoc logit adjustment) is chosen on VAL: max macro recall with val acc drop <= 2 points.
"""
import re
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig

from src.models.e3nn.spectral_model import SpectralModel
from src.scripts.train import GPULoader, get_dataset, make_top_density_transform

TAUS = (0.0, 0.25, 0.5, 0.75)
MAX_ACC_DROP = 0.02


def best_ckpt(run_dir: Path) -> Path:
    cands = []
    for p in run_dir.glob("model-*.ckpt"):
        loss = re.search(r"val_loss=([0-9]+\.[0-9]+)", p.name)
        ep = re.search(r"epoch=(\d+)", p.name)
        if loss and ep:
            cands.append((float(loss.group(1)), -int(ep.group(1)), p))
    if not cands:
        raise FileNotFoundError(f"no model-*.ckpt in {run_dir}")
    return min(cands, key=lambda t: (t[0], t[1]))[2]


def load_model(ck: Path, device):
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    for k in ("kw", "kwargs"):  # in case Lightning kept var-keyword args nested
        if isinstance(hp.get(k), dict):
            hp.update(hp.pop(k))
    model = SpectralModel(**hp)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def collect_probs(models, loader):
    per_model = [[] for _ in models]
    ys = []
    for x, y in loader:
        for i, m in enumerate(models):
            per_model[i].append(F.softmax(m(x).float(), dim=-1))
        ys.append(y)
    return [torch.cat(p) for p in per_model], torch.cat(ys)


def macro_recall(pred, y, num_classes):
    metric = torchmetrics.Recall(
        num_classes=num_classes, average="macro", task="multiclass", zero_division=0
    ).to(pred.device)
    return metric(pred, y).item()


def evaluate(probs, y, log_prior, tau, num_classes):
    pred = (probs.clamp_min(1e-12).log() - tau * log_prior).argmax(dim=-1)
    return (pred == y).float().mean().item(), macro_recall(pred, y, num_classes)
def full_report(probs, y, log_prior, tau, num_classes, title=""):
    from sklearn.metrics import classification_report

    pred = (probs.clamp_min(1e-12).log() - tau * log_prior).argmax(dim=-1)
    print(f"\n=== {title} (tau={tau:.2f}) ===")
    print(
        classification_report(
            y.cpu().numpy(), pred.cpu().numpy(), digits=4, zero_division=0
        )
    )
    k = min(5, num_classes)
    top5 = (probs.topk(k, dim=-1).indices == y[:, None]).any(-1).float().mean().item()
    print(f"top-{k} acc: {top5:.4f}")

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    runs = list(cfg.runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg.train.out_channels

    data_dir = Path(cfg.paths.data_dir)
    all_classes = sorted(
        {
            d.name
            for folder_name in ["train", "val", "test"]
            for d in (data_dir / folder_name).iterdir()
            if d.is_dir()
        }
    )
    label_to_idx = {name: idx for idx, name in enumerate(all_classes)}
    transform = make_top_density_transform(cfg.model.get("max_points", 2000))
    bs, nw = cfg.machine.batch_size, cfg.machine.num_workers
    val_loader = GPULoader(get_dataset(data_dir / "val", cfg, transform, label_to_idx), bs, False, device, nw)
    test_loader = GPULoader(get_dataset(data_dir / "test", cfg, transform, label_to_idx), bs, False, device, nw)

    names, models = [], []
    for r in runs:
        try:
            ck = best_ckpt(Path(cfg.paths.model_checkpoint) / r)
            models.append(load_model(ck, device))
            names.append(r)
            print(f"[{r}] loaded {ck.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[{r}] SKIPPED: {type(e).__name__}: {str(e)[:150]}")
    if not models:
        raise SystemExit("no runs could be loaded")
    log_prior = models[0].log_prior

    val_probs, val_y = collect_probs(models, val_loader)
    test_probs, test_y = collect_probs(models, test_loader)

    print("\nIndividual runs (tau=0):")
    for r, vp, tp in zip(names, val_probs, test_probs):
        va, vr = evaluate(vp, val_y, log_prior, 0.0, num_classes)
        ta, tr = evaluate(tp, test_y, log_prior, 0.0, num_classes)
        print(f"  {r:28s} val acc {va:.4f} rec {vr:.4f} | test acc {ta:.4f} rec {tr:.4f}")

    ens_val = torch.stack(val_probs).mean(0)
    ens_test = torch.stack(test_probs).mean(0)

    print(f"\nEnsemble of {len(models)} runs:")
    print("  tau   val_acc  val_rec | test_acc test_rec")
    rows = []
    for t in TAUS:
        va, vr = evaluate(ens_val, val_y, log_prior, t, num_classes)
        ta, tr = evaluate(ens_test, test_y, log_prior, t, num_classes)
        rows.append((t, va, vr, ta, tr))
        print(f"  {t:4.2f}  {va:.4f}   {vr:.4f}  | {ta:.4f}   {tr:.4f}")

    full_report(ens_val, val_y, log_prior, t, num_classes, title="Ensemble (VAL)")
    full_report(ens_test, test_y, log_prior, t, num_classes, title="Ensemble (TEST)")

    # pick max val acc; ties go to the smaller tau
    t, va, vr, ta, tr = max(rows, key=lambda r: (r[1], -r[0]))
    print(
        f"\nTau chosen on VAL: {t:.2f}  ->  test acc {ta:.4f}, test macro recall {tr:.4f}"
    )


if __name__ == "__main__":
    main()